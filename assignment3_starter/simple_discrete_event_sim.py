from __future__ import annotations
import heapq
import itertools
from typing import Any, Optional, Tuple
import random
"""Dijkstra implementation (baseline) -- awaiting dataset"""

class WeightedGraph: 
    def __init__(self):
        self.adjList_edges = {}
    
    def addNode(self,node):
        if (node in self.adjList_edges):
            print('Node in list') 
        else:
            self.adjList_edges[node] = []
    def addEdge(self,node1, node2, weight):
        if (node1 not in self.adjList_edges or node2 not in self.adjList_edges):
            print("One of the nodes is not in the list")
        else: 
           self.adjList_edges[node1].append((node2, weight))
           #self.adjList_edges[node2].append((node1, weight))
    def modifyWeight(self,node1, node2, weight):
        if (node1 not in self.adjList_edges or node2 not in self.adjList_edges):
            print("One of the nodes is not in the list")
        else:
            for i in range(len(self.adjList_edges[node1])):
                if (self.adjList_edges[node1][i][0] == node2):
                    self.adjList_edges[node1][i] = (node2,weight)
            for i in range(len(self.adjList_edges[node2])):
                if (self.adjList_edges[node2][i][0] == node1):
                    self.adjList_edges[node2][i] = (node1,weight)
    def getNeighbors(self,node):
        if (node not in self.adjList_edges):
            return ("Node is not in the list")
        else: 
            return(self.adjList_edges[node])
    def getNodes(self):
        return self.adjList_edges.keys()
    def __str__(self):
        return f"{self.adjList_edges}"

    def dijkstra_shortest_path(self,start_node, end_node): 
        '''
        @param: start node, end node
        @return: a tuple containing a list of visited elements, total cost 
        '''
        if (start_node not in self.adjList_edges or end_node not in self.adjList_edges): 
            return ("One of the nodes does not exist... Must add first")
        if (start_node == end_node): 
            return ([start_node], 0)
        
        distances = {}
        for node in self.adjList_edges.keys():
            distances[node] = float('inf')

        distances[start_node] = 0
        
        pred = {} #predecessor that store the node we used to get to our current node
        for node in self.adjList_edges.keys():
            pred[node] = None
        visited = []
        pq = PriorityQueue()

        for node in self.adjList_edges.keys():
            pq.insert(node, distances[node]) #distances = priority
        
        while not pq.isEmpty():
            min_tuple = pq.extractMin()
            if min_tuple is None:
                break
            
            current_node, current_distance = min_tuple
            
            #avoids repetitions
            if current_node in visited:
                continue
            visited.append(current_node)

            if current_node == end_node:
                break

            neighbors = self.adjList_edges[current_node]

            for neighb, cost in neighbors:
                if neighb not in visited:
                    #calculates the new distances
                    newDist = distances[current_node] + cost
                    #updates the cost of current node 
                    if newDist < distances[neighb]:
                        distances[neighb] = newDist
                        pred[neighb] = current_node
                        pq.insert(neighb, newDist)
        if distances[end_node] == float('inf'):
            return([],float('inf'))
        path = []
        i = end_node
        # starts from the end node and iterates back to get the path
        while i is not None:
            path.insert(0,i)
            i = pred[i]
        return(path,distances[end_node])   
    

class PriorityQueue: 
    #elements = [(item,priority), (item2,priority2),....]
    def __init__(self):
        self.elements = []
    def heapifyUp(self,nodeIndex):
        if (nodeIndex > 0 ): 
            tempNode = self.elements[nodeIndex]
            parentIndex = int((nodeIndex-1)/2)
            parentNode = self.elements[parentIndex]
            if (parentNode[1] > tempNode[1]): 
                # Swap temp node with its parent
                temp = tempNode
                self.elements[nodeIndex] = parentNode
                self.elements[parentIndex] = temp
                self.heapifyUp(parentIndex)

    def heapifyDown(self, nodeIndex):
        """
        Moves a node down the heap to maintain min-heap property.
        A node is swapped with its smaller child until it's in the correct position.
        """
        smallest = nodeIndex
        leftChildIndex = 2 * nodeIndex + 1
        rightChildIndex = 2 * nodeIndex + 2
    
        # Check if left child exists and is smaller than current node
        if leftChildIndex < len(self.elements):
            if self.elements[leftChildIndex][1] < self.elements[smallest][1]:
                smallest = leftChildIndex
    
        # Check if right child exists and is smaller than current smallest
        if rightChildIndex < len(self.elements):
            if self.elements[rightChildIndex][1] < self.elements[smallest][1]:
                smallest = rightChildIndex
    
        # If smallest is not the current node, swap and continue heapifying
        if smallest != nodeIndex:
            self.elements[nodeIndex], self.elements[smallest] = \
                self.elements[smallest], self.elements[nodeIndex]
            self.heapifyDown(smallest)
    '''
    Adds an item to the queue with an associated priority.
    '''
    def insert(self,item, priority):
        self.elements.append((item,priority))
        self.heapifyUp(len(self.elements)-1) 
    '''
    Removes the item with the minimum priority from the queue and returns it. If the
    queue is empty, this can return None.
    '''
    def extractMin(self):
        if(self.isEmpty()):
            return
        min = self.elements[0]
        self.elements[0] = self.elements[len(self.elements)-1]
        del self.elements[-1]
        if (not self.isEmpty()):
            self.heapifyDown(0)
        return min      
    '''
    Finds the item in the queue and updates its priority to this
    new value provided,priority. You can assume that the new priority 
    will always be less than the item’s current priority. Note that 
    your code should place the item to its correct position after the change.'''
    def decreaseKey(self,item, priority):
        for i, (it, p) in enumerate(self.elements):
            if(it == item):
                self.elements[i] = (item, priority)
                self.heapifyUp(i)
                return

    '''
    Returns True if the priority queue is empty, and False otherwise.
    '''
    def isEmpty(self):
        return len(self.elements) == 0
    def __str__(self):
        return f"{self.elements}"
#keep track of the statistics in the Car object
#at time t how many cars have used that edge 
class Car:
    def __init__(self, id, startNode, currentNode, endNode , dij_path, dij_cost):
        self.id = id
        #self.start_node = startNode
        #self.end_node = endNode
        #self.current = currentNode
        self.dij_path = dij_path
        self.dij_cost = dij_cost
        self.sim_cost = 0
        self.path_with_times = []
        self.arrival_time = 0
        self.completion_time = 0
        self.path_index = 0 
    def __str__(self):
        return f"Car(id={self.id}, start={self.start}, current={self.current}, end={self.end}, dij_path={self.dij_path}, dij_cost ={self.dij_cost}, sim_cost ={self.sim_cost})"
    
def read_graph(fname):
    # Open the file
    file = open(fname, "r")
    # Read the first line that contains the number of vertices
    # numVertices is the number of vertices in the graph (n)
    numVertices = file.readline()
    wg = WeightedGraph()

    # You might need to add some code here to set up your graph object
    # Next, read the edges and build the graph
    for line in file:
        # edge is a list of 3 indices representing a pair of adjacent vertices and the weight
        # edge[0] contains the first vertex (index between 0 and numVertices-1)
        # edge[1] contains the second vertex (index between 0 and numVertices-1)
        # edge[2] contains the weight of the edge (a positive integer)
        edge = line.strip().split(",")
    # Use the edge information to populate your graph object
    # TODO: Add your code here
        if (int(edge[0]) not in wg.getNodes()):
            wg.addNode(int(edge[0]))
        if (int(edge[1]) not in wg.getNodes()):
            wg.addNode(int(edge[1]))
        wg.addEdge(int(edge[0]),int(edge[1]),int(edge[2]))
    # Close the file safely after done reading
    file.close()
    return wg 
"""
Reads the agents file and stores them in a list of Car objects
"""
def read_agents(fname):
    # Open the file
    file = open(fname, "r")
    # Set up your list of agents
    agents=[]
    id = 1
    # Next, read the agents and build the list
    for line in file:
        # agent is a list of 2 indices representing a pair of vertices
        # path[0] contains the start location (index between 0 and numVertices-1)
        # path[1] contains the destination location (index between 0 and numVertices-1)
        path = line.strip().split(",")
        car = Car(id, int(path[0]), None, int(path[1]), [], 0)
        agents.append(car)
        id = id + 1
    # Close the file safely after done reading
    file.close()
    return agents


"""
simple_discrete_event_sim.py

A minimal discrete-event simulator where scheduled events carry an
event_id and optional payload. Event behavior is implemented by
overriding the Simulator.handle(event_id, payload) method using a
simple switch (if/elif) inside it.
"""
"""
you cant run sim for both algorithms at the same time..
two code files two classes for baseline and algorithm X
put flags in the code and decide which algorithm 
"""
class EventHandle:
    """Simple cancelable handle for a scheduled event."""
    __slots__ = ("_cancelled",)
    def __init__(self) -> None:
        self._cancelled = False
    def cancel(self) -> None:
        self._cancelled = True
    @property
    def cancelled(self) -> bool:
        return self._cancelled

class Simulator:
    """Minimal discrete-event simulator.
    Subclass and override handle(event_id, payload) with a switch-case.
    """
    def __init__(self, start_time: float = 0.0) -> None:
        self.now = float(start_time)
        self._queue: list[Tuple[float, int, Any, Any, EventHandle]] = []
        self._seq = itertools.count()
        self._stopped = False
        self.events_processed = 0

    def schedule_at(self, time: float, event_id: Any, payload: Any = None) -> EventHandle:
        if time < self.now:
            raise ValueError("Cannot schedule in the past")
        seq = next(self._seq)
        h = EventHandle()
        heapq.heappush(self._queue, (float(time), seq, event_id, payload, h))
        return h

    def _pop_next(self):
        while self._queue:
            time, seq, event_id, payload, h = heapq.heappop(self._queue)
            if not h.cancelled:
                return time, event_id, payload
            # skipped cancelled
        return None

    def step(self) -> bool:
        if self._stopped:
            return False
        item = self._pop_next()
        if item is None:
            return False
        time, event_id, payload = item
        self.now = time
        # dispatch to user-defined handler
        self.handle(event_id, payload)
        self.events_processed += 1
        return True

    def run(self, until: Optional[float] = None, max_events: Optional[int] = None) -> None:
        self._stopped = False
        processed = 0
        while not self._stopped:
            if not self._queue:
                break
            if until is not None and self._queue[0][0] > until:
                break
            if max_events is not None and processed >= max_events:
                break
            if not self.step():
                break
            processed += 1

    def stop(self) -> None:
        self._stopped = True

    def handle(self, event_id: Any, payload: Any) -> None:
        """Override in a subclass with a simple switch (if/elif) on event_id."""
        raise NotImplementedError("Override handle(event_id, payload)")



# Example usage with a simple switch-case style handler
if __name__ == "__main__":
    # simple simpulator that extends the main simulator
    class SimpleSim(Simulator):
        def __init__(self):
            super().__init__()
            self.graph = self.graph.adj #stores graph information at every step
            self.edge_congestion = {} #edge: (car_id, start_time, end_time)
            self.cars_state = {} #records where cars are at at each step of the sim
        def handle(self, event_id: str, payload: Any) -> None:
            # simple switch-case implemented with if/elif
            if event_id == "A":
                print(f"[{self.now:.3f}] {payload}")
                

            elif event_id == "D":
                print(f"[{self.now:.3f}] departure")
                # reschedule recurring heartbeat
                self.schedule_at(self.now+ 1.0, "departure", payload)
            elif event_id == "E":
                print(f"[{self.now:.3f}] stopping")
                self.stop()
            else:
                print(f"[{self.now:.3f}] unknown event {event_id!r} -> {payload}")
        
        def handle_arrival(self, payload):
            car_id = payload["id"]
            start = payload["start"]
            end = payload["end"]
            dij_path=payload["dij_path"]
            if dij_path == []: #if no path exists for car
                print(f"No path found for Car {car_id}")
                return
                
            self.cars_state[car_id] = {"path": dij_path, "position":0}
            print(f"[{self.now:.3f}] Car {car_id} enters system, path = {dij_path}")
            self.move_next_edge(car_id)

        def move_next_edge(self, car_id):
            state = self.cars_state[car_id]
            path = state["path"]
            curr = state["position"]

            if curr>= len(path) -1: #if curr has finished iterating through
                self.schedule_at(self.now, "car_done", {"car_id":car_id})
                return

            u = path[i]
            v=path[i+1]
            base_weight = self.graph[u][v]

            k=0 
            edge = (u,v)
            if edge in self.edge_congestion:
                for traffic in self.edge_congestion[edge]:
                    car, start, end = traffic
                    if start<=self.now<end: #if the car is currently using the edge
                        k+=1
            congestion_offset = k*base_weight
            global_cost += congestion_offset

            if edge not in self.edge_congestion:
                self.edge_congestion[edge] =[]
                self.edge_congestion.append =([car_id,self.now,self.now + congestion_offset])
            print(f"Car {car_id} starts {u}, {v}: base weight = {base_weight}, k = {k} ,total congestion = {congestion_offset}")

            #Schedule departure event to queue
            self.schedule_at(self.now + congestion_offset, "D", {"car_id": car_id, "edge": (u, v)})
        
        def handle_departure(self, payload):
            car_id = payload["car_id"]
            u, v = payload["edge"]

            #Remove car from congestion at edge
            edge = (u,v)
            if edge in self.edge_congestion:
                updated = []
                for data in self.edge_congestion[edge]:
                    if data[0] != car_id:
                        updated.append(data)
                self.edge_congestion[edge] = updated
            
            self.cars_state[car_id]["position"] +=1 #car officially leaves next edge and traverses next node in path

            print(f"[{self.now:.3f}] Car {car_id} leaves edge {u},{v} ")
            self.move_next_edge(car_id) #Car continues on path on graph


    sim = SimpleSim()
    cars = read_agents("input/agents16.txt")
    graph = read_graph("input/grid16.txt")
    k_at_t = {}

    #sets the k (number of cars at every edge) to zero at beginning of simulation
    for node, value in graph.adjList_edges.items():
        for node2, weight in value: 
            k_at_t[(node,node2)] = 0
            sim.edge_cost[(node,node2)] = weight
    
    #scheduled the arrival for every car
    for car in cars: 
        dij_result= graph.dijkstra_shortest_path(car.start, car.end) #Run dijkstra for each car
        car.arrival_time = 0
        car.dij_path = dij_result[0]
        car.dij_cost = dij_result[1]
        rand = random.randint(0,20) #Random number generator
        sim.schedule_at(rand, "A" , car)
    
    # sim.schedule_at(1,"A","c1")
    # sim.schedule_at(1,"A", "c2")
    # sim.schedule_at(1,"A", "c3")
    # sim.schedule_at(1,"D", "c4") 
    
    #for every event on the queue 
    for i in len(sim._queue):
        #logic for general 
        if sim._queue[i]:
            
    # while sim._queue: #changed to for loop for testing purposeds
    #     #print(sim._queue)
    #     first = sim._pop_next() #Get first item in queue
    #     print(first)
    #     if first: #If not null
    #         current_time = first[0]
    #         event_id = first[2]
    #         payload = first[3]
    #         sim.now = current_time

    #         simult_events = [(event_id,payload)] #track first item in simultaneous events 
    #         #simult_events = [(current_time, event_id,payload)] #track first item in simultaneous events 

    #         #stores all simultaneous events
    #         while sim._queue: #iterating to other events in queue
    #             print(f'this is supposed to be the next event{sim._queue}')
    #             next_time = sim._queue[0][0]
    #             if next_time == current_time:
    #                 next_event = sim._pop_next()
    #                 if next_event:
    #                     _, next_id, next_payload = next_event
    #                     simult_events.append((next_id, next_payload))
    #             else:
    #                 break
    #         for event_id, payload in simult_events:
    #             sim.handle(event_id, payload)



    print(f"Processing {len(simult_events)} events at time {current_time}")
    print(f"Events: {simult_events}")    
        # for id, pl in simult_events:
            #if id = "D" pop the event
            #if id = "A" check the next node for all cars 
                #for all cars that have the same next node update k value for edge 

    #Must pop all events that have the same time as the popped events
    print(f"sim now: {sim.now}")
    sim.run()
    sim._pop_next()
    print(f"sim now: {sim.now}")
    


    # sim.schedule_at(1.0, "say", "first at t=1.0") #how to add to schedule
    # h = sim.schedule_at(2.0, "say", "second at t=2.0 (will be canceled)")
    # sim.schedule_at(3.0, "say", "third at t=3.0")
    # #sim.now = sim._pop_next()[0]  --> How to update the simulation clock
    # print(sim.now)
    # h.cancel() #option to cancel events -- feature 
    # # schedule a stop event at t=2.2
    # sim.schedule_at(10, "stop", None)

    # sim.schedule_at(1.0, "say", "first at t=1.0") #how to add to schedule
    # h = sim.schedule_at(2.0, "say", "second at t=2.0 (will be canceled)")
    # sim.schedule_at(3.0, "say", "third at t=3.0")
    # h.cancel() #option to cancel events -- feature 
    # sim.schedule_at(0.5, "heartbeat", 0.5)
    # # schedule a stop event at t=2.2
    # sim.schedule_at(10, "stop", None)

    print("events processed:", sim.events_processed)
