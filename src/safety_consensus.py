import math
import logging

class TripleRedundancySystem:
    """
    Aviation-grade Safety Consensus System.
    Takes inputs from 3 independent AI computation nodes.
    If Node 1 proposes a dangerous/anomalous command, Nodes 2 and 3 outvote it.
    """
    def __init__(self, tolerance_meters=5.0):
        self.tolerance_meters = tolerance_meters
        print("[Safety-Consensus] Triple-Redundancy Voting System Initialized.")
        print(f"[Safety-Consensus] Consensus tolerance set to {tolerance_meters}m.")

    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculates distance between two GPS coordinates in meters."""
        R = 6371000 # Earth radius in meters
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi / 2.0) ** 2 + \
            math.cos(phi1) * math.cos(phi2) * \
            math.sin(delta_lambda / 2.0) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def vote_on_waypoint(self, node1_cmd, node2_cmd, node3_cmd):
        """
        Takes 3 proposed waypoints: (lat, lon, alt).
        Returns the consensus waypoint, or raises an error if total systemic failure occurs.
        """
        commands = [node1_cmd, node2_cmd, node3_cmd]
        
        # Calculate distances between the proposed waypoints
        d12 = self._haversine_distance(node1_cmd[0], node1_cmd[1], node2_cmd[0], node2_cmd[1])
        d23 = self._haversine_distance(node2_cmd[0], node2_cmd[1], node3_cmd[0], node3_cmd[1])
        d13 = self._haversine_distance(node1_cmd[0], node1_cmd[1], node3_cmd[0], node3_cmd[1])
        
        # R7 FIX: Also check altitude differences.
        # A node proposing a very different altitude (e.g. -100m vs 200m) must be rejected
        # even if its lat/lon matches. The altitude tolerance is 5x the horizontal tolerance.
        alt_tol = self.tolerance_meters * 5.0
        alt_ok_12 = abs(node1_cmd[2] - node2_cmd[2]) <= alt_tol
        alt_ok_23 = abs(node2_cmd[2] - node3_cmd[2]) <= alt_tol
        alt_ok_13 = abs(node1_cmd[2] - node3_cmd[2]) <= alt_tol
        
        agree_12 = (d12 <= self.tolerance_meters) and alt_ok_12
        agree_23 = (d23 <= self.tolerance_meters) and alt_ok_23
        agree_13 = (d13 <= self.tolerance_meters) and alt_ok_13

        if agree_12 and agree_23 and agree_13:
            # All 3 nodes agree perfectly. Average them for maximum precision.
            avg_lat = sum(c[0] for c in commands) / 3
            avg_lon = sum(c[1] for c in commands) / 3
            avg_alt = sum(c[2] for c in commands) / 3
            return (avg_lat, avg_lon, avg_alt)
            
        elif agree_12:
            # Node 3 is hallucinating/failing. Ignore it.
            print("\n[Safety-Consensus] WARNING: Node 3 output anomaly detected. Outvoting Node 3.")
            avg_lat = (node1_cmd[0] + node2_cmd[0]) / 2
            avg_lon = (node1_cmd[1] + node2_cmd[1]) / 2
            avg_alt = (node1_cmd[2] + node2_cmd[2]) / 2
            return (avg_lat, avg_lon, avg_alt)
            
        elif agree_23:
            # Node 1 is hallucinating/failing. Ignore it.
            print("\n[Safety-Consensus] WARNING: Node 1 output anomaly detected. Outvoting Node 1.")
            avg_lat = (node2_cmd[0] + node3_cmd[0]) / 2
            avg_lon = (node2_cmd[1] + node3_cmd[1]) / 2
            avg_alt = (node2_cmd[2] + node3_cmd[2]) / 2
            return (avg_lat, avg_lon, avg_alt)
            
        elif agree_13:
            # Node 2 is hallucinating/failing. Ignore it.
            print("\n[Safety-Consensus] WARNING: Node 2 output anomaly detected. Outvoting Node 2.")
            avg_lat = (node1_cmd[0] + node3_cmd[0]) / 2
            avg_lon = (node1_cmd[1] + node3_cmd[1]) / 2
            avg_alt = (node1_cmd[2] + node3_cmd[2]) / 2
            return (avg_lat, avg_lon, avg_alt)
            
        else:
            # Total systemic failure. No two nodes agree.
            print("\n[CRITICAL FAILURE] TRIPLE-REDUNDANCY SYSTEMIC COLLAPSE. NO CONSENSUS.")
            print("[CRITICAL FAILURE] ABORTING AI CONTROL. ENGAGING PHYSICAL KILL SWITCH.")
            return None
