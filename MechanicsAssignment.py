import datetime as dt
import pandas as pd
import utils_mgo
import vrp_algorithm

import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger().setLevel(logging.ERROR)


class MechanicsAssignment:
    def __init__(self, selected_date : dt.date = dt.datetime.today().date(),
                     appointments : pd.DataFrame = None,
                     cache : bool = True):
        """
        Initialize MechanicsScheduler class with selected date and options for caching data.
    
        Parameters:
        ----------
        selected_date : datetime.date, optional
            The date for which the scheduling is being done. Defaults to the current date if not provided.
        appointments : pd.DataFrame, optional
            DataFrame of appointments. If not provided, the function will load from file.
        cache : bool, optional
            Option to enable/disable caching of data for faster processing. Defaults to True.
        """
        
        self.selected_date = selected_date
        self.selected_sched = None
        
        self.cache = cache
        self.keys = None
        self.hubs_df = None
        
        self.mechanics = None
        self.mechanics_df = None

        self.appointments = appointments
        self.appointments_df = None # cleaned appointments
        
        self.data_loaded = False  # Track whether data has been loaded
        self.appointments_cleaned = False  # Track whether appointments have been cleaned
        self.lookback_days = 30
        
        if self.selected_date < dt.datetime.today().date():
            self.backtest = True
        
        else:
            self.backtest = False
        
        self.solution = None
        self.app_result = None
        
        self.load_config()
        self.load_data()  # Load initial data
        self.query_appointments()
        #self.optimize_assignment()
        
    
    def load_config(self):
        """
        Load configuration file for the service hubs, appointments key, and service duration.
        This function stores config data in self.keys for caching.
        """
        if self.cache and self.keys:
            return
        self.keys = utils_mgo.load_config()
    
    def load_data(self):
        """
        Load and prepare hubs, mechanics, and appointments data, independent of a selected date.
        This function caches the loaded data for future calls if caching is enabled.
        """
        if not self.data_loaded:
            # Load hubs data and prepare it
            if self.cache and self.hubs_df is None:
                self.hubs_df = utils_mgo.load_hubs_df(self.keys['service_hubs'])

            # Load mechanics data without filtering by date initially
            if self.cache and self.mechanics is None:
                self.mechanics = utils_mgo.load_mechanics_data()

            # Load appointments data
            if self.appointments is None:
                appointments = pd.read_csv(self.keys['appointments_key'])
                self.appointments = utils_mgo.clean_appointments(appointments,
                                                            self.mechanics,
                                                            self.keys['service_duration']) 
                
            # Flag that data has been loaded
            self.data_loaded = True
    
    def set_selected_date(self, new_selected_date: dt.date):
        """
        Update the selected_date without reloading or re-cleaning the data.
        The data will only be reloaded/cleaned when needed by other methods.

        Parameters:
        ----------
        new_selected_date : datetime.date
            The new selected date for appointment scheduling.
        """
        
        self.selected_date, self.selected_sched = utils_mgo.check_selected_date(new_selected_date)
        self.appointments_cleaned = False  # Mark that the appointments need to be cleaned
        self.data_loaded = False
        
        if self.selected_date < dt.datetime.today().date():
            self.backtest = False
        else:
            self.backtest = True
        
    
    def query_appointments(self):
        """
        Query appointments based on the selected date and clean the data.
        Only runs if appointments haven't already been cleaned for the selected date.

        Returns:
        ----------
        pd.DataFrame
            DataFrame containing cleaned and geocoded appointments.
        """
        if not self.appointments_cleaned:
            # Query appointments based on the selected date
            queried_appointments = utils_mgo.query_appointments(appointments = self.appointments,
                                                            selected_date= self.selected_date,
                                                            backtest = self.backtest)
            
            # Clean and geocode appointments
            cleaned_appointments = utils_mgo.clean_address_appts(queried_appointments)
            geocoded_appointments = utils_mgo.geocode(cleaned_appointments)

            # Update the cleaned appointments
            self.appointments_df = geocoded_appointments.sort_values('appointment_id').reset_index(drop = True)
            
            # filter for available mechanics
            self.mechanics_df = utils_mgo.available_mechanics(df=self.mechanics, 
                                                          selected_date=self.selected_date)
            
            # checks assigned hubs and mechanic availabilities
            self.assign_hub_service()
            self.check_mechanics_availability()
            
            # Mark as cleaned
            self.appointments_cleaned = True
            
    def calculate_distances(self):
        """
        Calculate the distances between the service hubs and the appointments.
        """
        appt_dist = utils_mgo.calculate_appointment_distances(self.appointments_df, 
                                                          self.hubs_df)
        return appt_dist

    def assign_hub_service(self):
        """
        Ensure that 'Hub Service' type appointments have an assigned hub.
        """
        appt_dist = self.calculate_distances()
        self.appointments_df = utils_mgo.check_hub_service(self.appointments_df, appt_dist)

    def check_mechanics_availability(self):
        """
        Check if the mechanics assigned to appointments are available and assign the correct hub.
        """
        self.appointments_df = utils_mgo.check_mechanic_availability(self.appointments_df, 
                                                                 self.mechanics_df)
    
    def get_assign_weights(self):
        lookback_date = self.selected_date - dt.timedelta(days = self.lookback_days)
        prev_appts = self.appointments[(self.appointments['appointment_date'] < self.selected_date) &
                                                           (self.appointments['appointment_date'] >= lookback_date)]
        
        assign_counts = {}
        for mechanic in self.mechanics_df['NAME'].unique():
            assign_counts[mechanic] = prev_appts['assigned_mechanic'].str.contains(mechanic).sum()
        
        # calculate weight penalties
        max_count = max(assign_counts.values()) if assign_counts else 1
        mechanic_weights = {mechanic : (max_count - count)/max_count for mechanic, count in assign_counts.items()}
        
        self.assign_weights = mechanic_weights
        
        return mechanic_weights
        
    def update_assignments(self):
        """
        Update the appointments DataFrame with the assigned mechanics and hubs from the solution DataFrame.
        """
        # Set appointment_id as index for both dataframes to match the rows
        self.appointments_df.set_index('appointment_id', inplace = True)
        
        # Use the update method to update assigned_mechanics and assigned_hub
        self.appointments_df.update(self.solution['assignments'])
        
        # Reset index to restore appointment_id as a column
        self.appointments_df.reset_index(inplace=True)
        self.appointments_df.rename(columns={'index':'appointment_id'}, 
                                                  inplace = True)
        
    def optimize_assignment(self):
        """
        Perform the optimization of mechanics assignments to appointments and return the solution.
        
        Returns:
        ----------
        solution_plan : dict
            The solution plan returned by the optimization function.
        """
        
        if self.selected_date is None:
            self.selected_date = dt.datetime.today().date()
            #raise Exception('selected_date is None.')
        
        if not self.appointments_cleaned:
            self.query_appointments()
        
        if self.backtest:
            self.appointments_df['mechanics'] = None
            self.appointments_df['assigned_mechanic'] = None
            self.appointments_df['assigned_hub'] = None
        
        self.assign_weights = self.get_assign_weights()
        
        # Perform optimization for mechanic assignment
        self.solution = vrp_algorithm.optimize_mechanics_assignment(self.mechanics_df,
                                                                    self.hubs_df,
                                                                    self.appointments_df,
                                                                    self.assign_weights,
                                                                    self.keys['tools'])
        # obtained a feasible solution
        if self.solution:
            # update assigned mechanics and hubs
            self.update_assignments()
            
            result = self.solution['assignments']
            result['appointment_date'] = self.selected_date.strftime('%Y-%m-%d')
            self.app_result = result.to_json(orient = 'records')
            
    def reset_data(self):
        '''
        Reset flags to relaod and reclean data.
        '''
        self.data_loaded = False
        self.appointments_cleaned = False
        self.appointments = None
        self.solution = None
        self.app_result = None
        
        self.load_data()
        self.query_appointments()
        self.assign_weights = self.get_assign_weights()
        

            
        