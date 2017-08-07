'''
(1) Which search algorithm seems to work best for each routing options?
- segments: BFS and ASTAR will give the optimal answer.
- distance: only ASTAR will give the best answer thanks to the heuristic function (haversine distance calculation, explained ahead).
- time: again, only ASTAR will give the optimal solution due to the heuristic function. The rest of the algorithms won't consider any particular path based on this
- scenic: if the optimal path turns out to be the one with less amount of highways, then BFS and ASTAR will give the same solution. If not, astar will give the optimal route. 

(2) Which algorithm is fastest in terms of the amount of computation time required by your program, and by how much, according to your experiments? (To measure time accurately, you may want to temporarily include a loop in your program that runs the routing a few hundred or thousand times.) 
In order to compare two algorithms, we ran the program 1000 times for each algorithm given a routing option. IDS and DFS performs very bad since these kind of search algorithms are not optimal for this particular problem, because of the way the graph is built. These algorithms  
take more than 10 times more computational time to cumpute wiht respect to BFS and ASTAR. 
In terms of finding the best route for the 'segments' option, if we compare ASTAR and BFS when we try to find the route from Bloomington,_Indiana to Chicago,_Illinois, BFS performs better than ASTAR:
 -> 1000 iterations of bfs take about 16.5 sec
 -> 1000 iterations of astar take about 31 sec
These are the only two comparissons we could do in terms of performance since for any other option, except for 'segments', ASTAR will actually perform acconding to the routing_option given by the user. 

(3) Which algorithm requires the least memory, and by how much, according to your experiments?
If we run the search algorithms to find a route from Bloomington,_Indiana to Chicago,_Illinois, we have the following results, given by the max size of the fringe:
BFS: 75
DFS: 1420
IDS: 1420
ASTAR: 77 (with routing option 'segments')
In this case DFS and IDS are the same because when IDS finds the goal state, will be at the same level and with the same fringe as DFS.

***These results may vary depending on the graph and the desired start_city and end_city.     

(4) Which heuristic function did you use, how good is it, and how might you make it better? 
For ASTAR, we have considered different heuristic functions based on the routing option. For 'segments' and 'scenic', we are only considering h to be = 0. But for 'distance', we are
calculating the haversine formula (https://en.wikipedia.org/wiki/Haversine_formula) to compute the distance of two points on a sphere from their latitude and longitude values, 
so f(n) = g(n) + h(n), where g(n) is the actual distance in miles to current city and h(n) is haversine distance to the goal. In cases where we don't have lat and lon values for a specific
city, we consider it to be = 0 (the junctions case). For 'time', we have consider h to be the same as before, but we divide that distance by a max_speed_limit value to get the "optimal" time 
considering a straight line between the two initial points. On way of improving the results would be considering something different to 0 when we don't have the lat and lon coordinates for some 
cities. 

(5) Supposing you start in Bloomington, which city should you travel to if you want to take the longest possible drive (in miles) that is still the shortest path to that city? (In other words, which city is furthest from Bloomington?)
In order to find this out, we need an algorithm that consider the cities in terms of the segment number but without a goal state. So in this case, this algorithm will search through all the cities with the least number of segments until 
the fringe is empty. To do this, we implemented Best First Search with no end_city. So, this algorithm will only save the solution with least number of segments at each iteration and will only finish once the fringe is empty, returning the
saved solution. 
If you want to run this algorithm, you can do it typing this on the command line: python route.py Bloomington,_Indiana _ segments bestfs     => Note that the second parameter can be anything. 
This algortihm will give tell us that we can go to Skagway,_Alaska from Bloomington with 54 "turns". 



Notes: We've decided not to influence the normal workflow of BFS, DFS and IDS. This means that they have a constant cost and the routing options are not considered for this algortihms because
they won't follow the normal functioning of them. When the nodes are ordered so that the one with the best evaluation is expanded first, we are applying best-first search.
We might end up in a path with nothing else than highways after we've made this decision. So, the only algorithm that contemplates
the routing options is ASTAR with its heuristic function in each case.
'''


import sys
import Queue
import time

# max speed limit assumed when calculating the gps distance and when speed limit is missing
max_speed_limit = 70


'''
Class containing the Route abstraction. Inside this class, all the information needed will be stored while finding the route.
'''
class Route:
    def __init__(self, cities_path, d, t, h, h_name):
        self.path = cities_path
        self.distance = d
        self.time = float(t)
        self.highway_path = h_name
        self.highways = h
        self.cost = 0

    def printItinerary(self):
        print 'We founded the best route based on your input options. You will arrive to your destination following these directions: \n'
        for x in xrange(0,len(self.path)-1):
            print 'From ', self.path[x].replace('_', ' ') , ' take highway ', self.highway_path[x]

        print ' '

'''
We created a dictionary containing the cities graph. Each key of the dictionary contains a unique city
and each item is a list which has the information of the cities that are connected to the key city.
Example: dic['Bloomington,_Indiana'] : [['Bedford,_Indiana', '21', '52', 'IN_37'], ['Cincinnati,_Indiana', '16', '45', 'IN_45'], ['Columbus,_Indiana', '32', '45', 'IN_46'], ['Martinsville,_Indiana', '19', '52', 'IN_37'], ['Spencer,_Indiana', '15', '45', 'IN_46']]
'''
def build_dict(path):
    
    road_segments_load = text_file = open(path, "r")
    road_segments_load = road_segments_load.readlines()
    road_seg = []
    for seg in road_segments_load:
        seg = seg.split(" ")
        seg[-1] = seg[-1].strip('\n')
        road_seg.append(seg)

    road_segments_dic = {}
    for city in road_seg:
        key1 = city[0]
        key2 = city[1]
        
        # just in case the speed limit is missing or it's equals to 0, set it to max_speed_limit.
        if city[3] == '':
            city[3] = max_speed_limit
        if city[3] == '0':
            city[3] = max_speed_limit
        # just in case the miles value is missing
        if city[2] == '':
            city[2] = '0'

        if key1 in road_segments_dic: 
            new_entry1 = road_segments_dic[key1] + [city[1:]]
        else: 
            new_entry1 = [city[1:]]
            
        if key2 in road_segments_dic:
            new_entry2 = road_segments_dic[key2] + [[city[0]]+city[2:]]
        else:
            new_entry2 = [[city[0]]+city[2:]]
            
        road_segments_dic[key1] = new_entry1
        road_segments_dic[key2] = new_entry2

    return road_segments_dic

'''
We created a dictionary containing the latitude and longitude for each city. Each key of the dictionary contains a unique city
and each item is a list which has the latitude and longitude information of that city.
Example: dic['Bloomington,_Indiana'] : ['39.165325', '-86.5263857']
'''
def getGpsCoord(path):
    city_gps = text_file = open(path, "r")
    city_gps = city_gps.readlines()
    gps_coord = []
    for seg in city_gps:
        seg = seg.split(" ")
        seg[-1] = seg[-1].strip('\n')
        gps_coord.append(seg)
    
    gps_coord_dic = {}
    for city in gps_coord:
        key = city[0]
        new_entry = city[1:]
        gps_coord_dic[key] = new_entry

    return gps_coord_dic

'''
The successors function returns a list of Route objects with its respective path
'''
def successors(init_city, routing_option, dic, visited):
    connected_cities = dic[init_city.path[-1]]    
    succ = []
    path = []
    for c in connected_cities:
        if not c[0] in visited:
            path = init_city.path + [c[0]]
            dist = int(init_city.distance) + int(c[1])
            time = float(init_city.time) + float(c[1])/float(c[2])
            
            if c[2] >= '55': 
                h = int(init_city.highways) + 1
            else:
                h = int(init_city.highways) + 0
            
            h_path = init_city.highway_path + [c[3]]
            succ.append(Route(path, dist, time, h, h_path))
            visited.append(c[0])
    return succ
 

def is_goal(route, end_city):
    if end_city == route.path[-1]:
        return True
    else:
        return False

'''
Return a list of Route objects containing the information of the cities that are connected to 'start_city'
'''    
def getConnectedRoutes(start_city, dic, fringe):
    
    connected_cities = dic[start_city]
    for c in connected_cities:
        path = [start_city] + [c[0]]
        time  = float(c[1])/float(c[2])        
        if c[2] >= '55': 
            highways = 1 
        else: 
            highways = 0
        h_path = [c[3]]
        
        c_i = Route(path, c[1], time, highways, h_path)
        fringe.put(c_i) 

'''
BFS algortihm that receives as input the start city, the end city (goal),
the routing option, the dictionary that has the city graph. If there route was found, it will return a Route object 
with all the information of the entire route, else it will return False
'''
def bfs(start_city, end_city, routing_option, dic):

    fringe = Queue.Queue()
    
    getConnectedRoutes(start_city, dic, fringe)
    
    visited = []
    visited.append(start_city)
    
    fringe_len = 0

    while not fringe.empty():
        curr_route = fringe.get()
        visited.append(curr_route.path[-1])
    
        if is_goal(curr_route, end_city):
            print fringe_len
            return curr_route
    
        for s in successors( curr_route, routing_option, dic, visited):
            if is_goal(s, end_city):
                print fringe_len
                return(s)
            fringe.put(s)

        if fringe.qsize() > fringe_len:
            fringe_len = fringe.qsize()

    return False

'''
DFS algortihm that receives as input the start city, the end city (goal),
the routing option, the dictionary that has the city graph, and the depth we would like 
to explore (just used when IDS is prefered, otherwise it will consider a depth equals to
the len of the number of different cities in the dictionary x10 to make sure that we will explore the whole graph.
'''
def dfs(start_city, end_city, routing_option, dic, depth):
    # connected_cities = dic[start_city]
    fringe = Queue.LifoQueue()

    getConnectedRoutes(start_city, dic, fringe)    
    
    visited = []
    visited.append(start_city)
    count = 0
    if depth == -1:
        depth = len(dic)*10

    fringe_len = 0

    while not fringe.empty() and count <= depth:
        curr_route = fringe.get()
        visited.append(curr_route.path[-1])
        
        if is_goal(curr_route, end_city):
            print fringe_len
            return curr_route
        
        for s in successors( curr_route, routing_option, dic, visited):
            if is_goal(s, end_city):
                print fringe_len
                return(s)
            fringe.put(s)
        count += 1

        if fringe.qsize() > fringe_len:
            fringe_len = fringe.qsize()

    return False

'''
IDS algortihm that receives as input the start city, the end city (goal),
the routing option, the dictionary that has the city graph, and the depth we would like to explore.
It calls the dfs algorithm with a hard-coded depht.
'''
def ids(start_city, end_city, routing_option, dic, depth):
    for i in range(0, depth):
        solution = dfs(start_city, end_city, routing_option, dic, i)
        if solution != False:
            return solution
    return False    

'''
Code extracted from http://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
Used to calculate the distance between two gps coordinates.
'''
from math import radians, cos, sin, asin, sqrt
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 3956 # Radius of earth in miles
    return c * r

'''
Heuristic function used for astar. It considers all possible routing options. When it is not possible to find gps coordinates in gpd dictionary, 
the heuristic function will equals to 0.
'''
def heuristic(curr_city, end_city, routing_option, gps):
    if routing_option == 'distance':
        if curr_city[0] in gps:
            lat_c_i = gps[curr_city[0]][0]
            lon_c_i = gps[curr_city[0]][1]
            h = haversine(curr_city[0], end_city)
        else:
            h = 0
    if routing_option == 'time':
        if curr_city[0] in gps:
            lat_c_i = gps[curr_city[0]][0]
            lon_c_i = gps[curr_city[0]][1]
            h = haversine(curr_city[0], end_city)/max_speed_limit
        else:
            h = 0
    return h

'''
Astar algortihm that receives as input the start city, the end city (goal),
the routing option, the dictionary that has the city graph, and the dictionary that has the 
gps locations for each city
'''
def astar(start_city, end_city, routing_option, dic, gps):

    fringe = Queue.PriorityQueue()
    fringe_len = 0
    # check if current state is goal state    
    if start_city == end_city:
        print fringe_len
        return Route([start_city], 0, 0, 0)

    connected_cities = dic[start_city]
    if end_city in gps:
        lat_end_city = gps[end_city][0]
        lon_end_city = gps[end_city][1]

    for city_i in connected_cities:
        path = [start_city] + [city_i[0]]
        time  = float(city_i[1])/float(city_i[2])
        if city_i[2] >= '55': 
            h = 1 
        else: 
            h = 0
        h_path = [city_i[3]]
        
        route_i = Route(path, city_i[1], time, h, h_path)

        if routing_option == 'segments':
            priority = len(route_i.path)
        if routing_option == 'scenic':
            priority = route_i.highways
        if routing_option == 'distance':
            h = heuristic(route_i.path[-1], end_city, routing_option, gps)
            priority = h + int(route_i.distance)
        if routing_option == 'time':
            h = heuristic(route_i.path[-1], end_city, routing_option, gps)
            priority = h + float(route_i.time)

        fringe.put((priority, route_i))


    visited = []
    visited.append(start_city)
    
    while not fringe.empty():
        curr_route = fringe.get()[1]
        visited.append(curr_route.path[-1])
        
        if is_goal(curr_route, end_city):
            print fringe_len
            return curr_route
        
        for s in successors( curr_route, routing_option, dic, visited):
            if is_goal(s, end_city):
                print fringe_len
                return(s)
            if routing_option == 'segments':
                priority = len(s.path)
            if routing_option == 'scenic':
                priority = s.highways
            if routing_option == 'distance':
                h = heuristic(s.path[-1], end_city, routing_option, gps)
                priority = h + int(s.distance)
            if routing_option == 'time':
                h = heuristic(s.path[-1], end_city, routing_option, gps)
                priority = h + float(s.time)
            
            fringe.put((priority,s))
            visited.append(s.path[-1])

        if fringe.qsize() > fringe_len:
            fringe_len = fringe.qsize()

    return False


'''
Best first search algortihm that receives as input the start city, the routing option, the dictionary that has the city graph, and the dictionary that has the 
gps locations for each city
'''
def bestfs(start_city, routing_option, dic, gps):

    fringe = Queue.PriorityQueue()

    connected_cities = dic[start_city]

    for city_i in connected_cities:
        path = [start_city] + [city_i[0]]
        time  = float(city_i[1])/float(city_i[2])
        if city_i[2] >= '55': 
            h = 1 
        else: 
            h = 0
        h_path = [city_i[3]]
        
        route_i = Route(path, city_i[1], time, h, h_path)

        priority = len(route_i.path)

        fringe.put((priority, route_i))


    visited = []
    visited.append(start_city)
    
    max_dist = 0

    while not fringe.empty():
        curr_route = fringe.get()[1]
        visited.append(curr_route.path[-1])
        
        for s in successors( curr_route, routing_option, dic, visited):
            
            if s.distance > max_dist:
                max_dist = s.distance
                sol = s

            priority = len(s.path)
            
            fringe.put((priority, s))
            visited.append(s.path[-1])

    return sol


def main(): 
    start_city = sys.argv[1] 
    end_city = sys.argv[2] 
    routing_option = sys.argv[3] 
    routing_algorithm = sys.argv[4]
    
    routing_options = ['segments', 'time', 'distance', 'scenic']
    routing_algorithms = ['bfs', 'dfs', 'ids', 'astar', 'bestfs']
    
    if not routing_option in routing_options:
        print 'Wrong routing option. Please choose one from: ', routing_options
        return
    
    if not routing_algorithm in routing_algorithms:
        print 'Wrong routing algorithm. Please choose one from: ', routing_algorithms
        return


    graph = build_dict("road-segments.txt")
    gps = getGpsCoord("city-gps.txt")


    # ####### Question 3: Computational time experiments 
    # start = time.time()
    # for x in xrange(1,1000):
    #     if routing_algorithm == 'bfs':
    #         sol = bfs(start_city, end_city, routing_option, graph)
    #     if routing_algorithm == 'dfs':
    #         sol = dfs(start_city, end_city, routing_option, graph, -1)
    #     if routing_algorithm == 'ids':
    #         sol = ids(start_city, end_city, routing_option, graph, len(graph)*10)
    #     if routing_algorithm == 'astar':
    #         sol = astar(start_city, end_city, routing_option, graph, gps)
    # end = time.time()
    # print(end - start)
    # #######


    if routing_algorithm == 'bfs':
        print 'Finding route with BFS...'
        sol = bfs(start_city, end_city, routing_option, graph)
    elif routing_algorithm == 'dfs':
        print 'Finding route with DFS...'
        sol = dfs(start_city, end_city, routing_option, graph, -1)
    elif routing_algorithm == 'ids':
        print 'Finding route with IDS...'
        sol = ids(start_city, end_city, routing_option, graph, len(graph))
    elif routing_algorithm == 'astar':
        print 'Finding route with ASTAR...' 
        sol = astar(start_city, end_city, routing_option, graph, gps)
    elif routing_algorithm == 'bestfs':
        ###### Question 5: Furthest city from Bloomington
        print 'Finding route with BEST FIRST SEARCH...'
        sol = bestfs(start_city, 'segments', graph, gps)

    
    print ''
    if sol != False:
        sol.printItinerary()
        print 'Segments: ', len(sol.path)
        string = '' 
        for s in sol.path:
            string = string + ' ' + s
        print sol.distance, ' ', sol.time, ' ', string
    else:
        print 'Sorry, Route not found. :('

    
if __name__ == "__main__": main()

