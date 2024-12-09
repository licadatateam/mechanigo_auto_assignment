# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 16:11:13 2024

@author: carlo

Mechanic Assignment Optimization Using OR-Tools

This script aims to solve the vehicle routing problem (VRP) with time windows and capacity constraints, 
specifically for assigning mechanics to appointments, considering the following:
    - Mechanics' working hours (time windows).
    - Appointment time windows.
    - Skill-based eligibility for services.
    - Distance and travel time between hubs and appointments.
    - Maximum number of appointments per mechanic.
    - Optional lunch break constraints.
    
Key Functions:
---------------
1. `create_mechanic_time_windows`: Creates time windows for mechanics based on their working hours.
2. `create_time_windows`: Generates appointment time windows from a reference time.
3. `calculate_distance`: Computes travel distance between two locations using the haversine formula.
4. `create_distance_matrix`: Builds a matrix of distances between hubs and appointments.
5. `create_data_model`: Prepares all necessary data for the OR-Tools solver, including distance and time matrices, time windows, and demand constraints.
6. `get_solution`: Extracts and prints the solution (assigned routes, times) once the problem is solved.
7. `add_lunchbreak`: Adds optional lunch break constraints for mechanics.
8. `optimize_mechanics_assignment`: The main function that prepares data, sets up the OR-Tools solver, and solves the mechanic assignment problem.
"""

import pandas as pd
import numpy as np
import random
import datetime as dt
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from geopy.distance import geodesic

pd.set_option('future.no_silent_downcasting', True)

#------------------------------------------------------------------------------

# Convert datetime objects to the number of seconds since a reference point (e.g., start of the day)
def convert_dt_to_sec(start_datetime, reference_datetime):
    # Example reference datetime for the start of the day (e.g., 00:00 on the day of appointments)
    return int((start_datetime - reference_datetime).total_seconds())

def convert_ts_to_dt(ts : pd.Timestamp, 
                     selected_date : dt.date):
    t = dt.datetime.strptime(ts.time().strftime('%H:%M'), '%H:%M')
    t_ = t.replace(year = selected_date.year, 
                   month = selected_date.month,
                   day = selected_date.day)
    return t_

# Function to create time windows for mechanics based on their available schedule in datetime format
def create_mechanic_time_windows(mechanics_df, reference_datetime):
    """
    Creates time windows for mechanics based on their available schedule.

    Args:
        mechanics_df (pd.DataFrame): DataFrame containing mechanics' work schedules (start and end times).
        reference_datetime (datetime): Reference point to convert datetime to seconds.

    Returns:
        list of tuples: A list of time window tuples (start, end) in seconds for each mechanic.
    """
    
    mechanic_time_windows = []
    for _, mechanic in mechanics_df.iterrows():
        start_time = pd.to_datetime(mechanic['START_TIME'])
        end_time = pd.to_datetime(mechanic['END_TIME'])
        start_sec = convert_dt_to_sec(start_time, reference_datetime)
        end_sec = convert_dt_to_sec(end_time, reference_datetime)
        mechanic_time_windows.append((start_sec, end_sec))
    return mechanic_time_windows
    
# Function to create time windows for appointments, also using datetime objects
def create_appt_time_windows(appointments_df, reference_datetime,
                        arrival_grace_period = 30):
    
    """
    Creates time windows for mechanics based on their available schedule.

    Args:
        appointments_df (pd.DataFrame): DataFrame containing appointments' booking data
        reference_datetime (dt.date): Reference point to convert datetime to seconds.
        arrival_grace_period (int) : Number of minutes for which mechanics are allowed to arrive late at hub

    Returns:
        list of tuples: A list of time window tuples (start, end) in seconds for each mechanic.
    """
    
    time_windows = []
    for _, appointment in appointments_df.iterrows():
        appointment_time_start = convert_ts_to_dt(
            appointment['appointment_time_start'], reference_datetime)
        appointment_start_min = appointment_time_start
        appointment_start_max = appointment_time_start + \
            dt.timedelta(minutes=arrival_grace_period)
        start_sec = convert_dt_to_sec(
            appointment_start_min, reference_datetime)
        end_sec = convert_dt_to_sec(appointment_start_max, reference_datetime)
        time_windows.append((start_sec, end_sec))
    return time_windows

# Function to calculate travel distance (haversine distance)
def calculate_distance(loc1, loc2):
    return geodesic(loc1, loc2).kilometers

# Function to create distance matrix (based on hubs and appointments)
def create_distance_matrix(hubs, appointments):
    locations = hubs + appointments
    distance_matrix = np.zeros((len(locations), len(locations)))
    
    for i, loc1 in enumerate(locations):
        for j, loc2 in enumerate(locations):
            if i != j:
                
                distance_matrix[i][j] = calculate_distance(loc1, loc2)
    
    distance_matrix = np.round(distance_matrix).astype(int)
    
    return distance_matrix

def create_skill_matrix(mechanics_df, appointments_df, base_penalty = 10000):
    """Creates a matrix of mechanics' eligibility for each appointment based on their skills, 
    with penalties for missing specific skills."""
    
    try:
        base_penalty = int(base_penalty)
    except:
        pass
    
    assert isinstance(base_penalty, int), f"Skill matrix penalty {base_penalty} not int type."
    
    # Define penalty values for each skill category
    penalty_weights = {
        'PMS': int(1*base_penalty),   # Highest penalty for missing PMS skill
        'CAR_BUYING_ASSISTANCE' : int(1*base_penalty),
        'INITIAL_DIAGNOSIS' : int(0.5*base_penalty),
        #'ELECTRICAL': int(0.3*base_penalty),     # Medium penalty for missing ELECTRICAL skill
        #'PARTS_REPLACEMENT': int(0.5*base_penalty),  # Lower penalty for missing PARTS REPLACEMENT skill
    }

    # Extract package categories from appointments and process them
    packages = appointments_df['package_category'].apply(lambda x: ['_'.join(s.split(' ')).strip() for s in x.split(', ')])

    mech_series = []
    for p in packages:
        # Initialize a series of penalties with zero (no penalty for satisfying the skill)
        penalty_series = pd.Series(0, index=mechanics_df.index)

        # For each skill in the appointment, check mechanics' eligibility
        for skill in p:
            # Check if the skill is in the mechanics' dataframe and assign penalties where skill is missing
            if skill in penalty_weights:
                penalty_series += mechanics_df[skill].fillna(False).apply(lambda eligible: 0 if eligible else penalty_weights[skill])
        
        mech_series.append(penalty_series)

    # Return the penalty matrix as a NumPy array (lower values indicate better skill matches)
    return pd.concat(mech_series, axis=1).to_numpy()

def create_data_model(appointments_df : pd.DataFrame,
                      mechanics_df : pd.DataFrame,
                      hubs_df : pd.DataFrame,
                      assign_weights : dict,
                      tool_reqs : dict):
    
    """
    Creates the data model required by OR-Tools to solve the VRP problem.

    Args:
        appointments_df (pd.DataFrame): DataFrame with appointment details including times and locations.
        mechanics_df (pd.DataFrame): DataFrame with mechanics' details including availability and hub assignment.
        hubs_df (pd.DataFrame): DataFrame with hub locations.

    Returns:
        dict: A dictionary containing all necessary inputs for OR-Tools (distance matrix, time windows, etc.).
    """
    
    data = {}
    data['appointments_df'] = appointments_df
    data['mechanics_df'] = mechanics_df
    data['hubs_df'] = hubs_df
    data['assign_weights'] = assign_weights
    data['tool_reqs'] = tool_reqs
    data["num_mechanics"] = len(mechanics_df)
    
    # assertions for hub and appointment coordinates
    assert 'lat' in hubs_df.columns and 'long' in hubs_df.columns, "Missing 'lat' or 'long' in hubs_df"
    assert 'lat' in appointments_df.columns and 'long' in appointments_df.columns, "Missing 'lat' or 'long' in appointments_df"
    assert not hubs_df[['lat', 'long']].isnull().any().any(), "hubs_df contains NaN in 'lat' or 'long'"
    assert not appointments_df[['lat', 'long']].isnull().any().any(), "appointments_df contains NaN in 'lat' or 'long'"
        
    # Coordinates for hubs and appointments
    hubs = [(hub['lat'], hub['long']) for _, hub in hubs_df.iterrows()]
    appointments = [(app['lat'], app['long']) for _, app in appointments_df.iterrows()]
    
    # Create distance matrix in km
    num_hubs = len(hubs)
    num_appointments = len(appointments)
    distance_matrix = create_distance_matrix(hubs, appointments).astype(int)
        
    assert distance_matrix.shape == (num_hubs + num_appointments, num_hubs + num_appointments), \
    "Distance matrix dimensions do not match the number of hubs and appointments"
    # Create time matrix in seconds
    time_matrix = (distance_matrix*3600/40).astype(int) # assuming average speed of 30 km/h
    
    # Create demand matrix
    # 0 for each hub, 1 for each appointment node
    num_nodes = len(distance_matrix)
    demand_matrix = [0]*len(hubs) + [1]*(num_nodes-len(hubs))
    assert len(demand_matrix) == num_nodes, "Demand matrix length does not match the number of nodes"
    
    data["distance_matrix"] = distance_matrix
    data["time_matrix"] = time_matrix
    data["demand_matrix"] = demand_matrix
    
    # Create time windows for each appointment and mechanic
    reference_dt = pd.Timestamp(appointments_df.appointment_date.min())
    mechanic_time_windows = create_mechanic_time_windows(mechanics_df,
                                                         reference_datetime = reference_dt)
    appt_time_windows = create_appt_time_windows(appointments_df, 
                                            reference_datetime = reference_dt)
    
    assert len(mechanic_time_windows) == data["num_mechanics"], "Mechanic time windows mismatch"
    assert len(appt_time_windows) == len(appointments), "Appointment time windows mismatch"
    
    # Specify list for depot/hub index
    def get_depot_index(hub_name):
        matching_hub = hubs_df[hubs_df['hub_name'] == f"{hub_name}_HUB"]
        assert not matching_hub.empty, f"Hub '{hub_name}_HUB' not found in hubs_df"
        return matching_hub.index[0]
    depot_ndx = mechanics_df['HUB'].apply(get_depot_index).tolist()
    
    data["appt_time_windows"] = appt_time_windows
    data["mechanic_time_windows"] = mechanic_time_windows
    data["depot_ndx"] = depot_ndx
    
    skill_matrix = create_skill_matrix(mechanics_df, appointments_df)
    assert skill_matrix.shape == (data["num_mechanics"], len(appointments)), \
        "Skill matrix dimensions do not match the number of mechanics and appointments"
    data['skill_matrix'] = skill_matrix
    
    return data
    
def get_solution(data, manager, routing, solution, time_dimension, 
                 appointments_df, mechanics_df, hubs_df):
    """
    Extracts the optimized solution from the OR-Tools model and returns the route plan.

    Args:
        data (dict): Data model containing distance matrix, time windows, and other constraints.
        manager (RoutingIndexManager): OR-Tools index manager for node routing.
        routing (RoutingModel): OR-Tools routing model object.
        solution (Assignment): The solution generated by OR-Tools solver.
        time_dimension (RoutingDimension): Time dimension used for time window constraints.
        appointments_df (pd.DataFrame): DataFrame containing appointments.
        mechanics_df (pd.DataFrame): DataFrame containing mechanics.
        hubs_df (pd.DataFrame): DataFrame containing hub locations.

    Returns:
        dict: Route plan and assignment data extracted from the solution.
    """
    
    print(f"Objective: {solution.ObjectiveValue()}")
    
    total_time = 0  # To accumulate the total time of all routes
    reference_dt = pd.Timestamp(appointments_df.appointment_date.min())
    
    def calc_route_start_time(mechanic_id, solution, manager, 
                              time_dimension, time_matrix):
        # get mechanic index
        index = routing.Start(mechanic_id)
        
        # route start time based on solution
        solution_start_time = solution.Min(time_dimension.CumulVar(index))
        
        # get index of first appointment (if any)
        # appointment indices are offset by number of hubs
        # but indices of no appointments results in large indices
        first_appt_index = solution.Value(routing.NextVar(index))
        if first_appt_index <= len(data['time_matrix']):
            time_var = time_dimension.CumulVar(first_appt_index)
            appt_start_time = solution.Min(time_var) # appointment start time
            travel_time = time_matrix[manager.IndexToNode(index), 
                                      manager.IndexToNode(first_appt_index)]
            # subtract travel time from appointment start time
            calc_start_time =  appt_start_time - travel_time
        else:
            # if no appointment, set same as solution
            calc_start_time = solution_start_time
        
        return int(max(calc_start_time, solution_start_time))
    
    total_route_plan = []
    solution_dict = {}
    
    for mechanic_id in range(data["num_mechanics"]):
        index = routing.Start(mechanic_id)
        
        route_start_time = calc_route_start_time(mechanic_id, solution,
                                                 manager, time_dimension,
                                                 data['time_matrix'])
        mechanic_start_time = dt.timedelta(seconds = data['mechanic_time_windows'][mechanic_id][0])
        mechanic_name = mechanics_df.iloc[mechanic_id,:]['NAME']
        hub_name = hubs_df.iloc[manager.IndexToNode(routing.Start(mechanic_id)),:]['hub_name']
        
        plan_output = f"Route for mechanic {mechanic_id} ({mechanic_name}):\n"
        plan_output += f"Mechanic start time: {str(mechanic_start_time)[:-3]}\n"
        plan_output += f"Route start time: {str(dt.timedelta(seconds=route_start_time))[:-3]}"
        plan_output += f" @ {hub_name}\n"
        
        prev_index = index
        while not routing.IsEnd(index):
            prev_node_index = manager.IndexToNode(prev_index)
            node_index = manager.IndexToNode(index)
            time_var = time_dimension.CumulVar(index)
            
            # Check if it's an appointment (not a hub)
            if node_index >= len(hubs_df):
                appt_ndx = node_index - len(hubs_df)
                appt_start_time = solution.Min(time_var)
                appt_end_time = convert_dt_to_sec(appointments_df.iloc[appt_ndx]['appointment_time_end'], reference_dt)
                prod_time = (appt_end_time - appt_start_time)/60.0
                #appt_end_time = solution.Max(time_var)  # Include service duration
                
                # Print formatted appointment start/end times
                appointment_id = appointments_df.iloc[appt_ndx, :]['appointment_id']
                plan_output += (
                    f"Appointment {appt_ndx} ({appointment_id}) "
                    f"Start: {dt.timedelta(seconds=appt_start_time)} "
                    f"End: {dt.timedelta(seconds=appt_end_time)}\n"
                )
                
                solution_dict[appointment_id] = {'barangay' : appointments_df.loc[appointments_df['appointment_id'] == appointment_id, 'barangay'].values[0],
                                                 'municipality' : appointments_df.loc[appointments_df['appointment_id'] == appointment_id, 'municipality'].values[0],
                                                 'province' : appointments_df.loc[appointments_df['appointment_id'] == appointment_id, 'province'].values[0],
                                                 'assigned_mechanic' : mechanic_name,
                                                 'assigned_hub' : hub_name,
                                                 'appt_start_time' : str(dt.timedelta(seconds=appt_start_time)),
                                                 'appt_end_time' : str(dt.timedelta(seconds=appt_end_time)),
                                                 'productive_time' : prod_time,
                                                 'travel_min' : data['time_matrix'][prev_node_index][node_index]/60.0,
                                                 'travel_km' : data['distance_matrix'][prev_node_index][node_index],
                                                 'sales' : appointments_df.loc[appointments_df['appointment_id'] == appointment_id, 'sale_total'].values[0]}
                
            prev_index = index
            index = solution.Value(routing.NextVar(prev_index))
        
        # Get end time and compute the total route time
        time_var = time_dimension.CumulVar(index)
        route_end_time = solution.Min(time_var)
        if route_end_time == route_start_time:
            route_duration = 0
            mechanic_end_time = dt.timedelta(seconds = data['mechanic_time_windows'][mechanic_id][1])
            plan_output += f"Mechanic end time: {str(mechanic_end_time)[:-3]}"
            
        else:
            route_duration = (route_end_time - route_start_time) / 60  # Convert to minutes
            plan_output += f"Route end time: {str(dt.timedelta(seconds=route_end_time))[:-3]}"
        plan_output += f" @ {hubs_df.iloc[manager.IndexToNode(routing.End(mechanic_id)),:]['hub_name']}\n"
        plan_output += f"Total time of this route: {route_duration:.2f} mins\n"
        
        print(plan_output)
        
        # Accumulate route time to total time
        total_time += route_duration
        
        total_route_plan.append(plan_output)
    
    # Print total time of all routes
    print(f"Total time of all routes: {total_time:.2f} mins")
    
    if solution_dict:
        output = pd.DataFrame(solution_dict).T
    else:
        output = None
    
    return {'assignments' : output.reset_index().rename(columns={'index' : 'appointment_id'}),
            'route_plan' : '\n'.join(total_route_plan)}

def add_lunchbreak2(routing, manager, data):
    """
    Adds optional lunch breaks for mechanics with penalties if skipped.
    """
    # Define lunch break parameters
    lunch_start_early = int(11 * 3600)  # 11:00 AM in seconds
    lunch_start_late = int(14 * 3600)  # 2:00 PM in seconds
    lunch_duration = int(30 * 60)  # 30 minutes in seconds
    late_penalty = 50  # Penalty for skipping the lunch break

    time_dimension = routing.GetDimensionOrDie('Time')

    for mechanic_id in range(data['num_mechanics']):
        # Define start and end time variables for the lunch break
        lunch_start = routing.solver().IntVar(lunch_start_early, lunch_start_late, f'lunch_start_{mechanic_id}')
        lunch_end = routing.solver().IntVar(lunch_start_early + lunch_duration, 
                                            lunch_start_late + lunch_duration, f'lunch_end_{mechanic_id}')

        # Create an optional interval variable for the lunch break
        lunch_interval = routing.solver().FixedDurationIntervalVar(
            lunch_start, lunch_duration, lunch_end, f'lunch_interval_{mechanic_id}'
        )
             
        # Get start and end indices for the mechanic
        end_index = routing.End(mechanic_id)

        # Penalize skipping the lunch break
        routing.AddDisjunction([end_index], late_penalty * 1000)

        # Add soft upper bound to encourage early lunch scheduling
        time_dimension.SetCumulVarSoftUpperBound(routing.End(mechanic_id), lunch_end.Max(), late_penalty)

        # Add lunch interval to the finalizer to minimize its start time
        routing.AddVariableMinimizedByFinalizer(lunch_start)
        routing.AddVariableMinimizedByFinalizer(lunch_end)

    return routing


def add_lunch_break3(routing, manager, data):
    """
    Adds optional lunch breaks as nodes in the routing model.
    """
    # Define lunch break parameters
    lunch_start_early = int(11 * 3600)  # 11:00 AM in seconds
    lunch_start_late = int(14 * 3600)  # 2:00 PM in seconds
    lunch_duration = int(30 * 60)  # 30 minutes in seconds
    lunch_penalty = 1000  # Penalty for skipping the lunch break

    # Add lunch break nodes to the data structure
    num_mechanics = data['num_mechanics']
    num_original_nodes = len(data['time_matrix'])

    # Extend the time matrix and time windows to include lunch break nodes
    for mechanic_id in range(num_mechanics):
        # Add a row and column for the new node
        temp_matrix = np.append(data['time_matrix'], [[0]]*num_original_nodes, axis=1)
        temp_matrix = np.append(temp_matrix, [[0]*(num_original_nodes + 1)], axis=0)

        # Extend time windows
        data['appt_time_windows'].append((lunch_start_early, lunch_start_late))

        # Add the lunch break node to the routing model
        lunch_node_index = num_original_nodes + mechanic_id
        routing.AddDisjunction([manager.NodeToIndex(lunch_node_index)], lunch_penalty)

        # Set the time window for the lunch break node
        time_dimension = routing.GetDimensionOrDie('Time')
        lunch_node_index_internal = manager.NodeToIndex(lunch_node_index)
        time_dimension.CumulVar(lunch_node_index_internal).SetRange(lunch_start_early, lunch_start_late)

        # Set the service time for the lunch break
        time_dimension.SlackVar(lunch_node_index_internal).SetValue(lunch_duration)

    return routing


def add_lunchbreak(routing, manager, data):
    """
    Adds flexible lunch breaks for mechanics using SetBreakIntervalsOfVehicle.
    Lunch breaks are optional but penalized if skipped.
    """
    # Define lunch break parameters
    lunch_start_early = int(11 * 3600)  # 11:00 AM in seconds
    lunch_start_late = int(14 * 3600)  # 2:00 PM in seconds
    lunch_duration = int(30 * 60)  # 30 minutes in seconds
    late_penalty = 5  # Penalty for skipping the lunch break

    time_dimension = routing.GetDimensionOrDie('Time')

    
    for mechanic_id in range(data['num_mechanics']):
        # Define the break as an interval with a start window and fixed duration
        lunch_start = routing.solver().IntVar(lunch_start_early, lunch_start_late, f'lunch_start_{mechanic_id}')
        lunch_end = routing.solver().IntVar(lunch_start_early + lunch_duration, 
                                            lunch_start_late + lunch_duration, f'lunch_end_{mechanic_id}')

        lunch_break = [
            routing.solver().FixedDurationIntervalVar(
                lunch_start, lunch_duration, lunch_end, f'lunch_break_{mechanic_id}'
            )
        ]

        # Apply the break interval to the mechanic's route
        time_dimension.SetBreakIntervalsOfVehicle(
            breaks=lunch_break,
            vehicle=mechanic_id,
            node_visit_transits=[0] * len(data['appointments_df'])  # No transit times between nodes
        )

        # Penalize skipping the break
        end_node = routing.End(mechanic_id)
        routing.AddDisjunction(
            [end_node], late_penalty * 1000
        )

    return routing



def check_solution(assignments, data):
    
    #assignments = solution_plan['assignments']
    # check if all appointments were given assignments
    assigned_appts = len(assignments)
    assign_check = 'Pass' if assigned_appts == len(data['appointments_df']) else 'Fail'
    print (f'Assignments check: {assigned_appts}/{len(data["appointments_df"])} = {assign_check}')
    
    mech_merge = assignments[['appointment_id', 'assigned_mechanic', 'assigned_hub']].merge(data["mechanics_df"][['NAME', 'CAR_BUYING_ASSISTANCE']], 
                                                                            left_on = 'assigned_mechanic', 
                                                                            right_on = 'NAME')
    mech_merge = mech_merge.drop('NAME', axis=1)
    
    appts_merge = assignments[['appointment_id', 'assigned_mechanic']].merge(data["appointments_df"][['appointment_id', 'package_category']],
                                                                             on = 'appointment_id')
    
    mech_appts_merge = mech_merge.merge(appts_merge, on = ['appointment_id', 'assigned_mechanic'])
    
    # check if number of mechanics with CBA assigned are equal or less than
    # number of appts with CBA
    appts_CBA = mech_appts_merge[mech_appts_merge['package_category'].str.contains('CAR BUYING ASSISTANCE', na = False)]
    mech_CBA = appts_CBA[appts_CBA['CAR_BUYING_ASSISTANCE'] == True]['assigned_mechanic'].nunique()
    CBA_check = 'Pass' if mech_CBA <= len(appts_CBA) else 'Fail'
    print (f'CBA Check: {mech_CBA}/{len(appts_CBA)} = {CBA_check}')
    
    # tool check
    try:
        tool_CBA = appts_CBA.groupby('assigned_hub')['assigned_mechanic'].nunique().reset_index()
        tool_CBA.loc[:, 'capacity'] = tool_CBA.apply(lambda x: data["tool_reqs"]['CAR BUYING ASSISTANCE'][x['assigned_hub'].split('_')[0]], axis=1)
        tool_CBA.loc[:, 'status'] = tool_CBA.apply(lambda x: 'Pass' if x['assigned_mechanic'] <= x['capacity'] else 'Fail', axis=1)
        tool_report = ''
        for ndx, row in tool_CBA.iterrows():
            tool_report += f'{row["assigned_hub"]}: {row["assigned_mechanic"]}/{row["capacity"]} = {row["status"]}\n'
        tool_check = 'Fail' if (tool_CBA['status'] == 'Fail').sum() else 'Pass'
        
        print (f'Tool check = {(tool_CBA["status"]=="Pass").sum()}/{len(tool_CBA)} = {tool_check}')
        print (tool_report)
    except:
        pass

def get_arc_costs(routing, manager, solution, data):
    """
    Extracts arc costs for each vehicle and appointment.

    Args:
        routing: The routing model.
        manager: The routing index manager.
        solution: The solution object returned by the solver.
        data: The data dictionary containing relevant matrices.

    Returns:
        dict: A dictionary where keys are vehicle IDs and values are lists of tuples (from_node, to_node, cost).
    """
    arc_costs = {}

    for vehicle_id in range(routing.vehicles()):
        mechanic_name = data['mechanics_df'].iloc[vehicle_id]['NAME']
        index = routing.Start(vehicle_id)
        vehicle_costs = []

        while not routing.IsEnd(index):
            # Get the next node
            next_index = solution.Value(routing.NextVar(index))
            
            # Convert indices to nodes
            from_node = manager.IndexToNode(index)
            to_node = manager.IndexToNode(next_index)
            
            # Get the arc cost
            arc_cost = routing.GetArcCostForVehicle(index, next_index, vehicle_id)
            
            def convert_node(node):
                if node < len(data['hubs_df']):
                    hub_name = data["hubs_df"].iloc[node]["hub_name"]
                    return f'{node} : {hub_name}'
                else:
                    ndx = node - len(data["hubs_df"])
                    appt_id = data['appointments_df'].iloc[ndx]['appointment_id']
                    return f'{ndx} : {appt_id}'
            
            vehicle_costs.append((convert_node(from_node), 
                                  convert_node(to_node), 
                                  arc_cost))

            # Move to the next node
            index = next_index

        arc_costs[f'{vehicle_id} - {mechanic_name}'] = vehicle_costs

    return arc_costs

def get_arc_cost(data, manager, routing, node1, node2, mechanic):
    
    if 'HUB' in str(node1):
        n1 = data['hubs_df'][data['hubs_df'].hub_name == node1].index[0]
    else:
        n1 = data['appointments_df'][data['appointments_df']['appointment_id']==str(node1)].index[0] + len(data['hubs_df'])

    if 'HUB' in str(node2):
        n2 = data['hubs_df'][data['hubs_df'].hub_name == node2].index[0]
    else:
        n2 = data['appointments_df'][data['appointments_df']['appointment_id']==str(node2)].index[0] + len(data['hubs_df'])

    m = data['mechanics_df'][data['mechanics_df']['NAME'] == mechanic].index[0]
    
    n1_ndx = manager.NodeToIndex(n1)
    n2_ndx = manager.NodeToIndex(n2)
    arc_cost = routing.GetArcCostForVehicle(n1_ndx, n2_ndx, m)
    return arc_cost
    
def create_manager(data : dict):
    """
    Constructs pywrapcp.RoutingIndexManager from given 'data' dict input.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing necessary information for assignment algorithm.
        
    Returns:
    --------
    manager : pywrapcp.RoutingIndexManager
    
    """
    
    manager = pywrapcp.RoutingIndexManager(len(data["time_matrix"]), 
                                           data["num_mechanics"], 
                                           data["depot_ndx"],
                                           data["depot_ndx"])
    
    return manager
    

# Updated function to optimize mechanic assignments
def optimize_mechanics_assignment(mechanics_df : pd.DataFrame, 
                                  hubs_df : pd.DataFrame, 
                                  appointments_df : pd.DataFrame,
                                  assign_weights : dict = None,
                                  tool_reqs : dict = None,
                                  penalties : dict = None):
    """
       Optimize mechanic assignments for a set of customer appointments using 
       Google OR-Tools to solve a vehicle routing problem (VRP).
       
       This function performs route optimization for mechanics based on several constraints 
       including:
         - Travel time between hubs and appointments.
         - Mechanics' available working hours.
         - Time windows for appointments.
         - Mechanics' skill sets.
         - Minimizing unassigned appointments and travel time.
       
       It uses the OR-Tools Routing API to handle mechanics as vehicles and customer appointments
       as deliveries, considering factors like skill requirements and available schedules. 
    
       Parameters:
       -----------
       mechanics_df : pd.DataFrame
           DataFrame containing mechanics' details including working hours, skill sets, and hubs.
       hubs_df : pd.DataFrame
           DataFrame containing details of the hubs from which mechanics are dispatched.
       appointments_df : pd.DataFrame
           DataFrame containing details of customer appointments including location, time windows, 
           and required services.
       assign_weights : dict
           Dictionary containing the weights for each available mechanic based on distribution of recent
           assignments.
    
       Returns:
       --------
       solution_plan : dict
           Dictionary containing:
               - 'assignments': A DataFrame mapping appointments to assigned mechanics and hubs.
               - 'route_plan': A string representation of the optimized routes for each mechanic.
           If no solution is found, the function returns `None`.
    
       Workflow Overview:
       -------------------
       1. Create the data model for the routing problem, which includes:
           - Distance matrix between hubs and appointments.
           - Time matrix to account for travel times.
           - Time windows for both mechanics and appointments.
           - Depot indices for mechanics' hubs.
           - Mechanics' skill eligibility for appointments.
    
       2. Set up the OR-Tools Routing Manager and Model.
           - Register transit callbacks for time and distance between locations.
           - Apply skill-based constraints to ensure mechanics are assigned only appointments they are 
             qualified for.
           - Set time windows for both mechanics and appointments.
    
       3. Add penalty systems for overtime and unassigned appointments.
    
       4. Solve the routing problem using a metaheuristic search algorithm (Guided Local Search) 
          with a time limit and solution limit.
    
       5. Process the solution if found, printing detailed routes for each mechanic and assigning 
          appointments. If no solution is found, return `None`.

   """
    # Step 1: Create data model from input data
    data = create_data_model(appointments_df,
                             mechanics_df,
                             hubs_df,
                             assign_weights,
                             tool_reqs)
    
    # OR-Tools data manager setup
    manager = create_manager(data)
    
    # routing
    routing = pywrapcp.RoutingModel(manager)
    
    def time_callback(from_index, to_index):
        """Returns travel time (seconds) between two nodes (hubs or appointments)."""
        # Convert from routing variable Index to distance matrix NodeIndex
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        
        return int(data["time_matrix"][from_node][to_node])
   
    def distance_callback(from_index, to_index):
        """Returns distance (in km) between two nodes (used for penalizing excessive travel)."""
        # Return the distance between the two nodes.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]
    
    def hub_tool_usage_callback(to_index):
        # Map internal node index to external node index
        to_node = manager.IndexToNode(to_index)

        # Check if this is an appointment (not a hub)
        if to_node >= len(data['hubs_df']):
            appointment_index = to_node - len(data['hubs_df'])
            
            # Get the hub associated with this appointment
            package_cat = data['appointments_df'].iloc[appointment_index]['package_category']

            # If the appointment belongs to this hub and requires a tool, return 1
            if 'CAR BUYING ASSISTANCE' in str(package_cat):
                return 1
        # No tool usage for this node
        return 0

    
    def weighted_transit_callback(data, manager, mechanic_ndx, mechanic_name,
                                  tool_capacity):
        """
        Custom cost function that includes both the base cost (e.g., travel time) 
        and a weight-based penalty for assigning mechanics with more recent assignments.
        """
        def callback(from_index, to_index):
            # Base cost (e.g., travel time or distance)
            # use external index (IndexToNode)
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            base_cost = data['time_matrix'][from_node][to_node]
            if base_cost >= 1350:
                base_cost = base_cost * 25
            
            # Penalty based on mechanic's recent assignment weight
            weight_penalty = (1 - data['assign_weights'][mechanic_name]) * 1000  # Scale as needed
            
            # skill penalty
            num_hubs = max(data['depot_ndx']) + 1
            # remove offset by hubs
            if to_node < num_hubs:
                appointment_index = to_node
            else:
                appointment_index = to_node - num_hubs
                
            skill_penalty = data['skill_matrix'][mechanic_ndx][appointment_index]
            
            # tool capacity penalty
            appt_package_cat = data['appointments_df'].iloc[appointment_index]['package_category']
            tool_penalty = 0 if tool_capacity and ('CAR BUYING ASSISTANCE' in str(appt_package_cat)) else 20000
            
            # Total cost includes base cost + weight penalty
            return int(base_cost + weight_penalty + skill_penalty + tool_penalty)
        
        return callback
    
    # Register the custom callback in OR-Tools
    def weighted_routing_model(routing, data):
        """
        Sets up the routing model with the weighted cost function based on past assignments.
        """

        # Default assignment weights if not provided
        if data['assign_weights'] is None:
            data['assign_weights'] = {mechanic : 1 for mechanic in data['mechanics_df']['NAME'].unique()}
        
        # Register a custom cost function for each mechanic (vehicle) using their weights
        for mechanic_ndx in range(data['num_mechanics']):
            mechanic_name = data['mechanics_df'].iloc[mechanic_ndx]['NAME']
            hub = data['mechanics_df'].iloc[mechanic_ndx]['HUB']
            tool_capacity = data['tool_reqs']['CAR BUYING ASSISTANCE'][hub]
            
            assert mechanic_name in data['assign_weights'], f"Mechanic '{mechanic_name}' not found in assign_weights"
            
            # Register the cost function for this specific mechanic
            callback = weighted_transit_callback(data, manager, mechanic_ndx, 
                                                 mechanic_name, tool_capacity)
            
            callback_index = routing.RegisterTransitCallback(callback)
            # Set the cost of travel to be the custom weighted cost function for this mechanic
            routing.SetArcCostEvaluatorOfVehicle(callback_index, mechanic_ndx)
    
        return routing
    
    routing = weighted_routing_model(routing, data)

    # Register time callback for routing model
    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    
    # Step 3: Define the time dimension to track travel and appointment durations
    time = 'Time'
    routing.AddDimension(
        transit_callback_index,
        slack_max = int(dt.timedelta(hours=6).total_seconds()),  # Allow some waiting time between jobs
        capacity = int(dt.timedelta(hours=24).total_seconds()),  # Max time for mechanic working hours (from start of day)
        fix_start_cumul_to_zero=False,  # Not all mechanics need to start at the same time
        name = time
    )
    time_dimension = routing.GetDimensionOrDie(time)
    
    service_times = [0] * len(hubs_df)
    # Add time window constraints for appointments
    for i, time_window in enumerate(data["appt_time_windows"]):
        index = manager.NodeToIndex(i + len(hubs_df))  # Offset by number of hubs
        time_dimension.CumulVar(index).SetRange(int(time_window[0]), int(time_window[1]))
     
        # Add service duration (slack) at the appointment
        appointment_time = appointments_df.iloc[i, :][['appointment_time_start',
                                                      'appointment_time_end']]
        service_time = int((appointment_time['appointment_time_end'] -
                           appointment_time['appointment_time_start']).total_seconds())
        
        service_times.append(service_time)
        
        time_dimension.SlackVar(index).SetValue(service_time)
    
    
    # Step 4: Set working hours constraints for mechanics and penalties for exceeding max working hours
    penalty_per_sec = 1
    max_working_hours = 15 * 3600
    # Add time window constraints for mechanics (their working hours)
    for mechanic_id, time_window in enumerate(data["mechanic_time_windows"]):
        index = routing.Start(mechanic_id)
        time_dimension.CumulVar(index).SetRange(int(time_window[0]), int(time_window[1]))
    
        # Add penalty for overtime
        end_var = time_dimension.CumulVar(routing.End(mechanic_id))
        routing.AddVariableMinimizedByFinalizer(end_var)  # Minimize overall time usage

        # Penalty for overtime (beyond max working time)
        routing.solver().Add(end_var <= max_working_hours + (penalty_per_sec * (end_var - max_working_hours)))
    
    # Step 5: Set flexible lunch break schedule
    #routing = add_lunchbreak2(routing, manager, data)
    
    # Step 6: Add penalties for waiting time
    # This ensures that the solver tries to minimize the mechanic's waiting time, 
    # encouraging them to start their routes closer to when they are actually needed for the first appointment
    for i in range(data["num_mechanics"]):
        routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.Start(i)))
        routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.End(i)))
            
    # Step 8: Set tool contraints

    tool_usage_dimensions = []
    hub_tool_capacity = list(data['tool_reqs']['CAR BUYING ASSISTANCE'].values())
    # Create a callback for this hub
    callback_index = routing.RegisterUnaryTransitCallback(hub_tool_usage_callback)
    
    for hub_index in range(len(data['hubs_df'])):
        # Add a dimension to track tool usage for this hub
        dimension_name = f"ToolUsage_{data['hubs_df'].iloc[hub_index]['hub_name']}"
        routing.AddDimension(
            evaluator_index=callback_index,  # Callback for this hub
            slack_max=0, # slack allowed
            capacity=hub_tool_capacity[hub_index]+1,  # Tool capacity for this hub
            fix_start_cumul_to_zero=True,    # Start tool usage at 0
            name=dimension_name
        )
        
        # Store the dimension for further use
        tool_usage_dimension = routing.GetDimensionOrDie(dimension_name)
        tool_usage_dimensions.append(tool_usage_dimension)
    
        penalty = 80000  # Adjust penalty as needed
        # # Set soft upper bound for this hub
        hub_internal_index = manager.NodeToIndex(hub_index)
        tool_usage_dimension.SetCumulVarSoftUpperBound(hub_internal_index, 
                                                        max(0, hub_tool_capacity[hub_index]-1), 
                                                        penalty)
        
        print (f'{hub_index}|{hub_internal_index} - {dimension_name} - {hub_tool_capacity[hub_index]}')
        #routing.solver().Add(tool_usage_dimension.CumulVar(hub_internal_index) <= hub_tool_capacity[hub_index])
        
    # Step 8: Add high penalty for unassigned appointments
    unassigned_penalty = 250000  # A large penalty for not assigning an appointment
    for appointment_index in range(len(appointments_df)):
        index = manager.NodeToIndex(appointment_index + len(hubs_df))  # Offset by hubs
        routing.AddDisjunction([index], int(unassigned_penalty))
    
    
    # Step 9: Add demand constraints to ensure mechanics do not exceed their capacity (2 appointments max)
    def demand_callback(from_index):
        """Returns the demand for a node. Appointments have a demand of 1, hubs have 0."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data["demand_matrix"][from_node]
    
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        [3]*data["num_mechanics"],  # vehicle maximum capacities
        True,  # start cumul to zero
        "Capacity",
    )
    
    # Step 6: Solve the problem using Guided Local Search (metaheuristic search)
    try:
        # Solve the problem
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        #search_parameters.log_search = True  # Enable logging
        search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC)
        
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        # large neighborhood search limit
        search_parameters.lns_time_limit.seconds = 30  # 10 seconds for LNS search steps
        search_parameters.time_limit.seconds = 60 # Limit total computation time
        #search_parameters.solution_limit = 200000  # Limit the number of solutions explored
        
        solution = routing.SolveWithParameters(search_parameters)
        
    except:
        solution = None
    
    # Step 7: Process the solution, if found, otherwise return None
    if solution:
        # print solution plan
        solution_plan = get_solution(data, manager, routing, solution, time_dimension, appointments_df, mechanics_df, hubs_df)
        check_solution(solution_plan['assignments'], data)
        
    else:
        print("No solution found.")
        solution_plan = None
        
    return solution_plan

# 20000 -> 231940 = 0/2 tool constraint fail
# 30000 -> 231940 = 0/2 tool constraint fail
# 60000 -> 231940 = 0/2 tool constraint fail
# slack_max = 0 -> 231940 = 0/2 tool constraint fail
# tool_capacity penalty = 15000 -> 301843 = 1/2 fail
# tool_capacity penalty = 30000 -> 510904 = 0/2 fail
# tool_capacity penalty = 5000 -> 159492 = 0/2 fail
# tool_capacity penalty = 2500 -> 121037 = 0/2 fail
# tool_capacity penalty = 20000 -> 371843 = 1/2 fail
# hub_capacity penalty = 100000 -> 371843 = 1/2 fail
# hub_capacity penalty = 20000 -> 371940 = 0/2 fail
# hub_capacity penalty = 150000 -> 371843 = 1/2 fail

# 36606
# 36832