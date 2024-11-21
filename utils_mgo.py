# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 21:48:10 2024

@author: carlo
"""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
import json
import toml
import datetime as dt
from fuzzywuzzy import fuzz, process
import re

def remove_punctuation(s: str) -> str:
    """
    Remove all punctuation from the input string except periods (.) and commas (,).
    
    Args:
    -----
        s : str
            The input string to clean.

    Returns:
    --------
        str
            A string with all punctuation removed except for periods and commas.

    Examples:
    ---------
    >>> remove_punctuation("Hello! This is a test: please, remove #punctuation.")
    'Hello This is a test please remove punctuation.'
    """
    # Define punctuation to keep (periods and commas) and remove other punctuation
    allowed_punct = {'.'}
    s_ = ''.join([char for char in s if char.isalnum() or char in allowed_punct or char.isspace()])
 
    return s_

def load_config(config_file='config_new'):
    """
    Load configuration settings from either a Streamlit secrets object or a JSON file.

    This function first attempts to load configuration settings from the Streamlit `secrets` object 
    (if using Streamlit). If the `secrets` are not available, it falls back to loading the configuration 
    from a JSON file with the given `config_file` name.

    Args:
    -----
        config_file : str, optional
            The name of the JSON file (without extension) to load configuration from if `st.secrets` is not available.
            Defaults to 'config_new'.

    Returns:
    --------
        keys : dict
            The loaded configuration keys.

    Raises:
    -------
        FileNotFoundError: If the JSON configuration file is not found.
        json.JSONDecodeError: If the JSON configuration file cannot be parsed.
    
    Examples:
    ---------
    >>> load_config('test_config')
    Error occurred: [Errno 2] No such file or directory: 'test_config.json'
    {}

    >>> load_config()['service_hubs']['makati_hub']  # Uses default 'config_new.json' file
    {'lat': 14.5640785, 'long': 121.0113147}
    """
    
    try:
        # Loading from a JSON file
        try:
            with open(f'{config_file}.json') as json_file:
                keys = json.load(json_file)
        except Exception as e:
            print(f"Error occurred: {e}")
            keys = {}
        
    except:
        # # Attempt to load from Streamlit secrets
        import streamlit as st
        keys = st.secrets['config']  
        
    return keys

def check_selected_date(selected_date: (dt.date, str) = None) -> tuple[dt.date, str]:
    """
    Validates and converts the input `selected_date` into a Python `datetime.date` object. 
    It also derives the corresponding weekday schedule string.

    Parameters:
    -----------
    selected_date : datetime.date, str, or None (Optional)
        The input date to check and convert. It can be a `datetime.date` object, a string
        representing a date, or `None`. If it's a string, it will be converted to a 
        `datetime.date` object. If it's `None`, the current date is returned.

    Returns:
    --------
    tuple:
        - selected_date (datetime.date): A valid `datetime.date` object representing 
          the input date or today's date if the input was `None`.
        - selected_sched (str): The corresponding weekday schedule column string in the 
          format 'MON_SCHED', 'TUE_SCHED', etc.

    Raises:
    -------
    ValueError:
        If the string cannot be parsed as a date or if the input date is invalid.

    Example:
    --------
    >>> check_selected_date('2023-09-15')
    (datetime.date(2023, 9, 15), 'FRI_SCHED')

    >>> check_selected_date(dt.date(2023, 9, 15))
    (datetime.date(2023, 9, 15), 'FRI_SCHED')

    >>> check_selected_date(None)
    (datetime.date(...), '..._SCHED')  # Returns today's date

    >>> check_selected_date('invalid-date')  # Raises ValueError
    Traceback (most recent call last):
        ...
    ValueError: Unknown string format: invalid-date
    """
    
    # Case 1: If selected_date is a string, attempt to convert it to datetime.date
    if isinstance(selected_date, str):
        try:
            selected_date = pd.to_datetime(selected_date).date()  # Convert string to date
        except ValueError as e:
            raise ValueError(f"Unknown string format: {selected_date}") from e
    
    # Case 2: If selected_date is None
    elif selected_date is None:
        selected_date = None
    
    # Case 3: If selected_date is already a datetime.date object, no conversion needed
    elif isinstance(selected_date, dt.date):
        pass
    
    # Raise an error if selected_date is of an unsupported type
    else:
        raise TypeError(f"Invalid type for selected_date: {type(selected_date)}. Must be str, datetime.date, or None.")
    
    if isinstance(selected_date, dt.date):
        # Derive the corresponding weekday schedule column, e.g., 'MON_SCHED', 'TUE_SCHED'
        selected_sched = selected_date.strftime('%A').upper()[:3] + '_SCHED'
    else:
        selected_sched = None
    
    return selected_date, selected_sched

def convert_to_toml(data : dict,
                    save : bool = True,
                    filename : str = None):
    
    """
    Converts a dictionary to TOML format and optionally saves it to a file.

    Parameters:
    -----------
    data : dict
        The input dictionary that needs to be converted to TOML format.
        
    save : bool, optional, default=True
        If True, the TOML data will be saved to a file. If False, the function 
        will only return the TOML string without saving.
        
    filename : str, optional, default=None
        The name of the file to save the TOML data. If not provided and `save` 
        is True, the function will generate a filename in the format 
        "secrets_YYYY-MM-DD.toml" using the current date.

    Returns:
    --------
    toml_str : str
        The TOML string converted from the input dictionary.

    Exceptions:
    -----------
    If an error occurs during the conversion or file-saving process, the exception 
    will be caught, and the error message will be printed.

    Example:
    --------
    >>> import toml
    >>> data = {'name': 'John Doe', 'age': 30, 'active': True}
    >>> toml_str = convert_to_toml(data, save=False)
    name = "John Doe"
    age = 30
    active = true
    <BLANKLINE>
    >>> print(toml_str)
    name = "John Doe"
    age = 30
    active = true
    <BLANKLINE>
    >>> toml_str == toml.dumps(data)
    True
    
    This will print the TOML string, and if `save` is True, it will save the string 
    to 'config.toml'.
    """
    
    try:
        # converts dictionary to toml string
        toml_str = toml.dumps(data)
        print(toml_str)
        
        # check if conditional to save
        if save:
            # compose filename if not provided
            if filename is None:
                import datetime as dt
                date_today = dt.datetime.today().date().strftime('%Y-%m-%d')
                filename = f"secrets_{date_today}.toml"
            
            # saves data to toml file
            with open(filename, 'w') as toml_file:
                toml.dump(data, toml_file)
                
    except Exception as e:
        print(e)
    
    return toml_str
    
def load_hubs_df(hubs_dict : dict) -> pd.DataFrame:
    """
    

    Args:
    -----
        hubs_dict : dict
            Hubs dictionary containing hub_name, lat & long coordinates

    Returns
    -------
    df_hubs : pd.DataFrame
        Organized dataframe containing hub information
        
    Examples:
    >>> load_hubs_df(keys['service_hubs'])

    """
    df_hubs = pd.DataFrame(hubs_dict).T.reset_index()
    df_hubs = df_hubs.rename(columns = {'index' : 'hub_name'})
    df_hubs.loc[:, 'hub_name'] = df_hubs['hub_name'].apply(lambda x: x + '_HUB')
    return df_hubs

def extract_hub(entry: str) -> str or None:
    """
    Extract hub information from the given entry using fuzzy string matching.

    Args:
    -----
        entry : str 
            The string entry from which to extract hub information.

    Returns:
    --------
        res : str or None
            The extracted hub information, or None if no match is found.

    Raises:
    -------
        ValueError: If the entry is not a string.
    
    Examples:
    ---------
    >>> extract_hub("I am near the makati hub")
    'makati_hub'

    >>> extract_hub("close to bulacan hub")
    'bulacan_hub'

    >>> extract_hub("unknown location")
    """
    if not isinstance(entry, str):
        print("Input entry must be a string.")
        return None

    hubs = load_hubs_df(load_config()['service_hubs']) # Load available hubs from configuration

    try:
        # Fuzzy match the entry against the hub names
        match = process.extractOne(entry, hubs['hub_name'], 
                                   scorer=fuzz.partial_ratio, 
                                   score_cutoff=80)
    except Exception as e:
        print(f"Error occurred: {e}")
        match = None

    # Return the best matching hub, or None if no match is found
    return match[0] if match is not None else None


def available_mechanics(df: pd.DataFrame, 
                        selected_date: dt.date = None) -> pd.DataFrame:
    """
    Filters the mechanics based on their availability for a specific schedule and 
    their active status.

    Parameters:
    -----------
    df : pd.DataFrame
        A DataFrame containing mechanics' data, including schedules, services, and their 
        active status. It must have columns for the weekday schedules (e.g., 'MON_SCHED', 
        'TUE_SCHED'), the 'STATUS' of the mechanics, and their 'POSITION'.
    
    selected_date : dt.date
        The column name representing the selected day’s schedule (e.g., 'MON_SCHED') 
        which indicates whether mechanics are available for appointments.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing mechanics who are available on the selected day and meet 
        the filtering criteria (active status, regular position, and schedule availability).
    
    Raises:
    -------
    ValueError:
        If the input DataFrame does not contain the `selected_sched` column.
    Example:
    --------
    >>> available_mechanics(df_mechanics, 'MON_SCHED')
    Returns a DataFrame of available mechanics on Monday.

    """
    
    # Check if the input DataFrame is valid
    if df is None or df.empty:
        raise ValueError("The input mechanics DataFrame is empty or not provided.")
    
    if selected_date is not None:
        # calculate selected_sched from selected_date
        selected_date, selected_sched = check_selected_date(selected_date)
        
        # Ensure that the required `selected_sched` column exists in the DataFrame
        if selected_sched not in df.columns:
            raise ValueError(f"'{selected_sched}' not found in the DataFrame columns.")
        
        # Check for valid schedule information (not null)
        sched_cond = df[selected_sched].notnull()
    
    else:
        sched_cond = df.NAME.notnull()

    # Filter mechanics based on their active status, regular position, and availability on the selected day
    filtered_df = df[(df['STATUS'] == 'ACTIVE') & 
                     (df['POSITION'].isin(['REGULAR', 'TRAINEE',
                                           'PROBATIONARY'])) & 
                     (df['HUB'] != 'HERTZ') &
                     sched_cond]
    
    # If no mechanics are available after filtering, raise a warning
    if filtered_df.empty:
        raise Exception(f"No mechanics available for the schedule: {selected_sched}")
        
    # Calculate start and end times for each available mechanic
    else:
        if selected_date is not None:
            filtered_df[['START_TIME', 'END_TIME']] = filtered_df[selected_sched].apply(
                lambda x: pd.Series(extract_schedule_times(x, selected_date)))
    
    return filtered_df.reset_index(drop = True)


def check_mechanic_availability(appointments_df: pd.DataFrame, 
                                mechanics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Checks if the assigned mechanic for each appointment is available. If available, assigns the hub 
    of the mechanic to the 'assigned_hub' field. If not available, sets 'assigned_mechanic' to None.

    Parameters:
    -----------
    appointments_df : pd.DataFrame
        DataFrame containing appointment data, including 'appointment_id', 'appointment_time',
        and 'assigned_mechanic'.
        
    mechanics_df : pd.DataFrame
        DataFrame containing mechanics data, including 'name', 'available_times' (as datetime ranges), 
        and 'hub' (the hub where the mechanic is located).

    Returns:
    --------
    pd.DataFrame
        Updated `appointments_df` with the 'assigned_hub' field set to the correct hub if the mechanic
        is available, or 'assigned_mechanic' set to None if they are not.
    """
    
    # Iterate over each appointment
    for idx, row in appointments_df.iterrows():
        assigned_mechanic = row['assigned_mechanic']
        
        if pd.isna(assigned_mechanic) or (len(assigned_mechanic) == 0):
            continue  # No mechanic assigned, skip
        
        # Find the assigned mechanic in the mechanics_df
        mechanic_info = mechanics_df[mechanics_df['NAME'] == assigned_mechanic]
        
        if mechanic_info.empty:
            # If the mechanic is not found, set to None
            appointments_df.at[idx, 'assigned_mechanic'] = None
            appointments_df.at[idx, 'assigned_hub'] = None
            continue
        
        # Extract the mechanic's availability and hub
        start_time = mechanic_info['START_TIME'].values[0]  # assuming this is a datetime range or list
        hub = mechanic_info['HUB'].values[0]
        
        appointment_time = row['appointment_time_start']  # assuming this is a datetime
        
        # Check if the mechanic is available during the appointment time
        if appointment_time >= start_time:
            # Mechanic is available, assign the correct hub
            appointments_df.at[idx, 'assigned_hub'] = hub
        else:
            # Mechanic is not available, set assigned_mechanic to None
            appointments_df.at[idx, 'assigned_mechanic'] = None
            appointments_df.at[idx, 'assigned_hub'] = None
    
    return appointments_df

def calculate_appointment_distances(appointments_df: pd.DataFrame, 
                                    hubs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the Haversine distance between appointments and hub locations.

    Parameters:
    -----------
    appointments_df : pd.DataFrame
        A DataFrame containing appointment data, which must include 'appointment_id', 'lat', and 'long' columns 
        for appointment ID and the latitude and longitude of each appointment.

    hubs_df : pd.DataFrame
        A DataFrame containing hub data, which must include 'hub_name', 'lat', and 'long' columns 
        for hub names and their respective latitude and longitude.

    Returns:
    --------
    distances_df : pd.DataFrame
        A DataFrame where each row represents an appointment, and each column (except for 'appointment_id') 
        represents the distance (in kilometers) to a given hub.
    """
    
    # Convert hub coordinates (lat, long) to radians
    hubs_radians = np.radians(hubs_df[['lat', 'long']].to_numpy())
    
    # Convert appointment coordinates (lat, long) to radians
    appointments_radians = np.radians(appointments_df[['lat', 'long']].to_numpy())
    
    # Calculate Haversine distances between appointments and hubs (in radians)
    distances = haversine_distances(appointments_radians, hubs_radians)
    
    # Convert the distance from radians to kilometers
    distances_km = distances * 6371.0088  # Earth radius in kilometers
    
    # Convert the distance array to a DataFrame and label columns with hub names
    distances_df = pd.DataFrame(distances_km, columns=hubs_df['hub_name'])
    
    # Add the 'appointment_id' column for easier referencing
    distances_df['appointment_id'] = appointments_df['appointment_id'].values
    
    return distances_df

def check_hub_service(appointments_df: pd.DataFrame, 
                      appt_dist: pd.DataFrame) -> pd.DataFrame:
    """
    Assign the nearest hub to appointments that require hub services, based on distance.

    Parameters:
    -----------
    appointments_df : pd.DataFrame
        A DataFrame containing appointment data, which includes 'appointment_id' and 'transaction_type'.
        Rows where 'transaction_type' is 'Hub Service' will have a nearest hub assigned.
        
    appt_dist : pd.DataFrame
        A DataFrame containing the distances between appointments and hubs.
        Must include 'appointment_id' and columns representing distances to each hub (e.g., columns containing '_HUB').

    Returns:
    --------
    pd.DataFrame
        The updated `appointments_df` with a new column 'assigned_hub' indicating the closest hub for each hub service.
    """
    
    # Filter only appointments that are 'Hub Service'
    hub_service_filter = (appointments_df['transaction_type'] == 'Hub Service') & (appointments_df['assigned_hub'].isnull())
    
    # Extract the hub-related columns (those containing '_HUB')
    hub_cols = appt_dist.columns[appt_dist.columns.str.contains('_HUB')]
    
    # Find the nearest hub for all appointments by taking the minimum distance along hub columns
    nearest_hub = appt_dist.loc[:, hub_cols].idxmin(axis=1)
    
    # Assign the nearest hub to the relevant appointments
    appointments_df.loc[hub_service_filter, 'assigned_hub'] = appointments_df.loc[hub_service_filter, 'appointment_id'].map(
        dict(zip(appt_dist['appointment_id'], nearest_hub))
    )
    
    return appointments_df


def clean_appointments(appointments : pd.DataFrame,
                       mechanics_df : pd.DataFrame,
                       service_duration : dict) -> pd.DataFrame:
    
    # Calculate 'appointment_time_end' 
    def calc_time_end(time_start: dt.datetime, 
                      package_category: str) -> dt.datetime:
        """
        Calculate the end time for an appointment by adding the duration of each package 
        category service to the start time.

        Args:
        - time_start (datetime.datetime): The starting time of the appointment.
        - package_category (str): A string of comma-separated service categories (e.g., "PMS, CAR BUYING ASSISTANCE").

        Returns:
        - datetime.datetime: The calculated end time after adding the service durations.
        
        Assumptions:
        - The service_duration dictionary contains the duration of each service category as a 
          dictionary with 'HOURS' and 'MINUTES'.
        
        Raises:
        - ValueError: If a package category is not found in the service_duration dictionary.
        """
        # Split the package_category string into a list of categories
        categories = package_category.split(', ')
        
        # Initialize time_end as time_start for the first calculation
        time_end = time_start

        for category in categories:
            # Get the duration for the current category
            if category not in service_duration:
                raise ValueError(f"Service category '{category}' not found in the service_duration dictionary.")
            
            duration = service_duration[category]
            # Add the hours and minutes for the current category to the time_end
            time_end += dt.timedelta(hours=duration['HOURS'], minutes=duration['MINUTES'])
        
        return time_end

    def match_mechanics(mechanics_str: str, 
                        mechanics_df: pd.DataFrame, 
                        score_cutoff: int = 80) -> str:
        """
        Matches a comma-separated string of mechanics to the closest names in a given mechanics DataFrame using fuzzy matching.

        Parameters:
        -----------
        mechanics_str : str
            A comma-separated string of mechanic names to be matched.
            
        mechanics_df : pd.DataFrame
            A DataFrame that contains a 'NAME' column with mechanic names.
            
        score_cutoff : int, optional (default=80)
            The minimum score for a fuzzy match to be considered valid.

        Returns:
        --------
        str
            A string of matched mechanic names, joined by '/'.
            Returns an empty string if no matches are found.
        """
        # Split the input string and clean mechanic names
        mechanics = [mechanic.strip() for mechanic in mechanics_str.split(',')]
        
        # Get unique mechanic names from the DataFrame
        unique_mechanics = mechanics_df['NAME'].unique()
          
        # Initialize list to store matched mechanics
        matched_mechanics = []
        
        # Perform fuzzy matching for each mechanic
        for mechanic in mechanics:
            match = process.extractOne(mechanic, unique_mechanics, 
                                       score_cutoff=score_cutoff)
            if match:
                matched_mechanics.append(match[0])
            else:
                continue
        
        # Remove duplicates
        matched_mechanics = list(set(matched_mechanics))

        # Return matched mechanics as a string, separated by '/'
        return '/'.join(matched_mechanics) if matched_mechanics else ''
    
    try:
        # Convert 'date' and 'time' columns to datetime
        appointments['appointment_date'] = pd.to_datetime(appointments['date'], format = '%m/%d/%y').dt.date
        appointments['appointment_time_start'] = pd.to_datetime(appointments['time'], format = '%H:%M:%S')
        
        # default PMS category
        # executed first to remove null category values
        appointments.fillna({'package_category' : 'PMS'}, inplace = True)
        
        # calculate appointment time end from service_duration
        appointments['appointment_time_end'] = appointments.apply(lambda x: calc_time_end(x['appointment_time_start'],
                                                                                          x['package_category']), axis=1)
        
        # Format 'appointment_time_start' and 'appointment_time_end' as strings
        appointments['appointment_time_start'] = appointments.apply(lambda x: x['appointment_time_start'].replace(year = x['appointment_date'].year,
                                                                                                                  month = x['appointment_date'].month,
                                                                                                                  day = x['appointment_date'].day),
                                                                    axis = 1)
        appointments['appointment_time_end'] = appointments.apply(lambda x: x['appointment_time_end'].replace(year = x['appointment_date'].year,
                                                                                                              month = x['appointment_date'].month,
                                                                                                              day = x['appointment_date'].day),
                                                                    axis = 1)
        
        appointments['province'] = appointments['province'].str.title()
        appointments.fillna({'address':''}, inplace = True)
        
        # Match assigned mechanics (if any)
        appointments['assigned_mechanic'] = appointments['mechanics'].apply(lambda x: match_mechanics(x, mechanics_df)).replace('', None)
        
        #  Apply extract_hub to determine the hub based on the 'pin_address'
        appointments.loc[:, 'assigned_hub'] = appointments['pin_address'].apply(extract_hub)
        
        
        # Drop duplicates and reset index
        appointments = appointments.drop_duplicates(subset=['appointment_id']).reset_index(drop=True)

        # Drop unnecessary columns (if they exist)
        appointments = appointments.drop(columns=['date', 'time'], errors='ignore')

    except Exception as e:
        raise e
    
    return appointments.replace('', None)

def query_appointments(appointments : pd.DataFrame, 
                       selected_date : dt.date,
                       backtest : bool = False) -> pd.DataFrame:

    """
    Query and process appointment data from a CSV and filter it based on the selected date.

    Args:
    -----
        appointments : pd.DataFrame
            Appointments data.
        selected_date : datetime.date
            The date to filter appointments on.
        mechanics_df : pd.DataFrame
            Dataframe of all mechanics data
        service_duration : dict, optional
            dictionary of durations (in hours and mins) for each package category
        
    Returns:
    --------
        pd.DataFrame
            A DataFrame containing the filtered appointment data with added hub solver.

    Raises:
    -------
        Exception: If there are no appointments for the selected date or there is a processing error.

    Notes:
    ------
    - The function reads appointments data, converts date and time fields, calculates appointment end times, 
      and filters the appointments based on the selected date.
    - Hub information is added using the `extract_hub` function.
    - The appointments are deduplicated based on the `appointment_id`.

    """
    
    try:
              
        # Filter appointments based on the selected date
        filtered_appointments = appointments.loc[appointments['appointment_date']==selected_date]

        # Filter appointments by confirmation status
        if backtest:
            filtered_appointments = filtered_appointments[filtered_appointments['date_confirmed'].notnull()]
        else:
            filtered_appointments = filtered_appointments[filtered_appointments['date_confirmed'].notnull() &
                                                          filtered_appointments['date_completed'].isnull()]
        
        if filtered_appointments.empty:
            raise Exception(f"No appointments available for {selected_date.strftime('%Y-%m-%d')}")
              
    
        # Drop duplicates and reset index
        filtered_appointments = filtered_appointments.drop_duplicates(subset=['appointment_id']).reset_index(drop=True)

        # Drop unnecessary columns (if they exist)
        filtered_appointments = filtered_appointments.drop(columns=['date', 'time'], errors='ignore')
        
    except Exception as e:
        raise e
    
    return filtered_appointments.replace('', None)

def clean_address(row: pd.Series, 
                  address_col: str = 'pin_address', 
                  barangay_col: str = 'barangay',
                  municipality_col: str = 'municipality', 
                  province_col: str = 'province') -> tuple[str, str, str]:
    """
    Clean and format the address information in a given row of a DataFrame.

    Args:
    -----
        row : pd.Series
            The row containing address-related columns.
        address_col : str (Optional)
            Column name for the address. Defaults to 'pin_address'.
        barangay_col : str (Optional)
            Column name for the barangay. Defaults to 'barangay'.
        municipality_col : str (Optional)
            Column name for the municipality. Defaults to 'municipality'.
        province_col : str (Optional)
            Column name for the province. Defaults to 'province'.

    Returns:
    --------
        Tuple[str, str, str]: 
            - street_address : str - The extracted street address.
            - street : str - Extracted street name.
            - cleaned_address : str - Full cleaned address string.
    
    Examples:
    ---------
    >>> import pandas as pd
    >>> data = {'pin_address': '123 Main St, Brgy. Example, Municipality A, Province X', 'barangay': 'Example', 'municipality': 'Municipality A', 'province': 'Province X'}
    >>> row = pd.Series(data)
    >>> clean_address(row)
    ('123 Main St', 'Main St', '123 Main St in Example, Municipality A, Province X in Philippines')
    """
    
    # Extract and clean address components
    address = row[address_col] if row[address_col] else ''
    barangay = row[barangay_col] if row[barangay_col] else ''
    municipality = row[municipality_col] if row[municipality_col] else ''
    province = row[province_col] if row[province_col] else ''
    
    # Special handling for certain cases
    if province.lower() == 'metro manila' and municipality.lower() == 'quezon':
        municipality = 'Quezon City'
        
    # Remove specific patterns like "B.F. Homes"
    remove_pattern = re.compile(r'b\.?f\.? homes', re.IGNORECASE)
    address = re.sub(remove_pattern, '', address)
    
    # Attempt to extract street address by splitting at barangay, municipality, or province
    try:
        if barangay and barangay in address:
            match = re.search(f'((Brgy\.)\s+)?{barangay}', address, re.IGNORECASE)
            street_address = address[:match.span()[0]].strip() if match else address.split(barangay)[0].strip()
        elif municipality and municipality in address:
            street_address = address.split(municipality)[0].strip()
        elif province and province in address:
            street_address = address.split(province)[0].strip()
        else:
            street_address = address
    except Exception:
        street_address = address
    
    # Remove punctuations in street_address
    street_address = remove_punctuation(street_address)
    
    # Create cleaned full address
    cleaned_address = ', '.join([''.join([street_address, f" in {barangay}" if barangay else '']),
                       municipality if municipality else '',
                       province if province else '']) + ' in Philippines'
    cleaned_address = re.sub('( ,|\s{2,})', '', cleaned_address)
    
    # Extract street name using a regex pattern for streets
    street_pattern = r'(\b\w+\s+(St\.?|Street)\b)'
    match = re.search(street_pattern, street_address, re.IGNORECASE)
    street = match.group(1) if match else ''
    
    return street_address, street, cleaned_address

def clean_address_appts(appointments: pd.DataFrame) -> pd.DataFrame:
    """
    Clean address information for all rows in an appointments DataFrame.

    Args:
    -----
        appointments : pd.DataFrame
            A DataFrame containing appointment data with address-related columns.

    Returns:
    --------
        pd.DataFrame
            A DataFrame with additional columns for cleaned addresses.

    """
    _appointments = appointments.copy()

    try:
        # Apply the clean_address function row-wise and expand the result into multiple columns
        _appointments[['street_address', 'street', 'address_query']] = _appointments.apply(
            clean_address, axis=1, result_type='expand')
        
    except Exception as e:
        raise e

    return _appointments

def fill_table_with_geocoding(df: pd.DataFrame, 
                              address_col: str = 'address', 
                              lat_col: str = 'lat', 
                              long_col: str = 'long') -> pd.DataFrame:
    """
    Geocode missing latitude and longitude values in a DataFrame and fill in missing coordinates.

    This function:
    1. Identifies rows with missing 'lat' and 'long' values.
    2. Geocodes the addresses in those rows using the Geocoder library.
    3. Updates the DataFrame with the filled geocoded coordinates.

    Args:
    -----
        df (pd.DataFrame): 
            The DataFrame containing address information and potentially missing 'lat'/'long' values.
        address_col (str, optional): 
            The column name for addresses to geocode. Defaults to 'address'.
        lat_col (str, optional): 
            The column name for latitude values. Defaults to 'lat'.
        long_col (str, optional): 
            The column name for longitude values. Defaults to 'long'.
    
    Returns:
    --------
        pd.DataFrame: The input DataFrame with updated 'lat' and 'long' values.
            
    """
    
    # Create an instance of the Geocoder and initialize the driver
    from gmaps_geocode import Geocoder
    gmaps = Geocoder()
    
    try:
        gmaps.initialize_driver()
        # Filter rows where latitude and longitude are missing
        missing_coords_df = df[df[lat_col].isnull() | df[long_col].isnull()]
        
        # Geocode the missing coordinates using the Geocoder library
        if not missing_coords_df.empty:
            geocoded_df = gmaps.geocode_dataframe(missing_coords_df, address_col)
            
            # Update the original DataFrame with the geocoded results
            df.update(geocoded_df, overwrite=False)
        else:
             raise Exception("No missing latitude or longitude values to fill.")

    except Exception as e:
        raise (f"Geocoding error: {str(e)}")
    
    finally:
        # Close the Geocoder driver
        gmaps.close_driver()

    return df

def geocode(df : pd.DataFrame) -> pd.DataFrame:
    """
    Geocode addresses in a DataFrame by merging with barangay-level latitude 
    and longitude data from a Google Sheet. If latitude and longitude are 
    missing after merging, the geocoding process will fill in the gaps.

    Args:
    -----
    df : pd.DataFrame
        The input DataFrame containing appointment data with barangay and 
        municipality columns.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with added or filled latitude and longitude columns.

    Raises:
    -------
    Exception:
        If the Google Sheet cannot be loaded or merged correctly.
    
    """
    
    try:
        # Load configuration data (e.g., Google Sheet ID and sheet name)
        config = load_config()
        brgy_sheet_id = config['mgo_sheet']
        brgy_sheet_name = config['barangays_sheet_name']
        
        # Construct the Google Sheets URL to load barangay lat/long data
        url = f'https://docs.google.com/spreadsheets/d/{brgy_sheet_id}/gviz/tq?tqx=out:csv&sheet={brgy_sheet_name}'
        
        # Read the barangays data from the Google Sheet into a DataFrame
        df_brgys = pd.read_csv(url)
    except Exception as e:
        raise Exception(f"Error loading barangays data from Google Sheets: {str(e)}")
    
    try:
        # Select only relevant columns for merging to reduce memory usage
        brgys_cols = ['barangay', 'municipality', 'lat', 'long']
        
        # Merge the input DataFrame with barangay data on 'barangay' and 'municipality'
        df_merged = df.merge(df_brgys[brgys_cols], on=['barangay', 'municipality'], how='left')
    except Exception as e:
        raise Exception(f"Error merging DataFrames: {str(e)}")
    
    try:
        # Check if there are any missing lat/long values and fill them via geocoding
        if df_merged['lat'].isnull().sum() > 0 or df_merged['long'].isnull().sum() > 0:
            df_merged = fill_table_with_geocoding(df_merged)
        
        # Drop duplicates and reset index for clean output
        df_merged = df_merged.drop_duplicates().reset_index(drop=True)
        
    except Exception as e:
        raise Exception(f"Error during geocoding or data processing: {str(e)}")
    
    return df_merged

def load_mechanics_data(selected_date : dt.date = None) -> pd.DataFrame:
    """
    Load mechanics data from a Google Sheet into a pandas DataFrame.

    This function retrieves mechanics data from a Google Sheet using the provided
    sheet ID and sheet name from the configuration file. The function constructs
    the appropriate URL to read the data into a pandas DataFrame.

    Returns:
    --------
    pd.DataFrame:
        A DataFrame containing the mechanics data from the Google Sheet.

    Raises:
    -------
    Exception:
        If there is an issue loading configuration data or reading the mechanics 
        data from the Google Sheet.

    Examples:
    ---------
    >>> df_mechanics = load_mechanics_data()
    >>> print(df_mechanics.head())
    """
    
    try:
        # Load configuration data (e.g., Google Sheet ID and sheet name)
        config = load_config()
        mechs_sheet_id = config['mgo_sheet']
        mechs_sheet_name = config['mechanics_sheet_name']
        
        # Construct the Google Sheets URL to load barangay lat/long data
        url = f'https://docs.google.com/spreadsheets/d/{mechs_sheet_id}/gviz/tq?tqx=out:csv&sheet={mechs_sheet_name}'
        
        # Read the data into a DataFrame from the Google Sheet URL
        df_mechanics = pd.read_csv(url).replace('', None)
        
    except Exception as e:
        # Raise a specific error if the data loading fails
        raise Exception(f"Error loading mechanics data from Google Sheets: {str(e)}")
    
    return df_mechanics

    
def extract_schedule_times(selected_sched: str,
                           selected_date : dt.date):
    """
    Extract the start and end times from a schedule string of the format 'A:BC to X:YZ'.
    
    Args:
    -----
        selected_sched (str): The schedule string in the format "A:BC to X:YZ".
    
    Returns:
    --------
        tuple: A tuple containing the start and end times as `datetime.time` objects.
    """
        
    # Regex pattern to capture times
    pattern = r'(\d{1,2}:\d{2} [APM]{2}) to (\d{1,2}:\d{2} [APM]{2})'
    
    # Search for the pattern in the given schedule string
    match = re.search(pattern, selected_sched)
    
    if match:
        start_time_str, end_time_str = match.groups()
        
        # Convert to datetime objects (optional)
        start_time = dt.datetime.strptime(start_time_str, '%I:%M %p')
        end_time = dt.datetime.strptime(end_time_str, '%I:%M %p')
        
        # Replace the date component with selected_date
        start_time = start_time.replace(year=selected_date.year, month=selected_date.month, day=selected_date.day)
        end_time = end_time.replace(year=selected_date.year, month=selected_date.month, day=selected_date.day)
        
        
        return start_time, end_time
    else:
        return None, None

def get_date_from_id(appointment_id : str or int,
                      appointments : pd.DataFrame):
    '''
    Extract appointment_date given appointment_id
    '''
    appointment_date = appointments.loc[appointments.appointment_id == str(appointment_id), 
                                        'appointment_date'].min()
    
    return appointment_date


def count_assignments(mechanics_df: pd.DataFrame, 
                      appointments: pd.DataFrame, 
                      start_date: dt.date = dt.date(2024, 1, 1)) -> pd.DataFrame:
    """
    Counts the number of assigned appointments per mechanic after a specified start date.
    This version handles cases where multiple mechanics are assigned to an appointment and are stored
    in a single string field (comma-separated).
    
    Parameters:
    ----------
    mechanics_df : pd.DataFrame
        DataFrame containing the list of mechanics with at least a 'NAME' column.
    
    appointments : pd.DataFrame
        DataFrame of appointments, containing at least 'appointment_date' and 'assigned_mechanic' columns.
        The 'assigned_mechanic' column can have multiple mechanic names separated by commas.
    
    start_date : datetime.date, optional
        The date from which to count assignments. Default is January 1, 2024.

    Returns:
    -------
    pd.DataFrame
        A DataFrame with two columns: 'mechanic_name' (name of the mechanic) and 'num_assignments' 
        (the number of appointments assigned to each mechanic after the start date).
    """
    
    # Filter appointments based on the start date and non-null mechanic assignments
    filtered_appointments = appointments[(appointments['appointment_date'] >= start_date) &
                                         (appointments['assigned_mechanic'].notnull())]
    
    # Initialize a dictionary to store the assignment counts for each mechanic
    assignments_dict = {mechanic: 0 for mechanic in mechanics_df['NAME'].unique()}
    
    # Iterate over each mechanic and count their assignments by checking if their name is in 'assigned_mechanic'
    for mechanic in assignments_dict.keys():
        # Count appointments where the mechanic's name is in the 'assigned_mechanic' field
        appt_slice = filtered_appointments[filtered_appointments['assigned_mechanic'].str.contains(mechanic, regex=False, na=False)]
        num_appts = len(appt_slice['appointment_id'].unique())  # Count unique appointments
        assignments_dict[mechanic] = num_appts
    
    # Create a DataFrame to return the results
    result_df = pd.DataFrame({
        'mechanic_name': list(assignments_dict.keys()),
        'num_assignments': list(assignments_dict.values())
    })
    
    return result_df
      