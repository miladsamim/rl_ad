"""
    Scenario to drive straight
"""

import random
import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorTransformSetter,
                                                                      StopVehicle,
                                                                      LaneChange,
                                                                      WaypointFollower,
                                                                      Idle)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import InTriggerDistanceToVehicle, StandStill, InTriggerDistanceToLocation
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.scenario_helper import get_waypoint_in_distance


class StraightDriving(BasicScenario):
    """"""

    timeout = 120
    
    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=600):
        self.timeout = timeout
        self._map = CarlaDataProvider.get_map()
        self._reference_waypoint = self._map.get_waypoint(config.trigger_points[0].location)
        self._distance = 100
        super(StraightDriving, self).__init__("Straight",
                                        ego_vehicles,
                                        config,
                                        world,
                                        debug_mode,
                                        criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        end_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._distance)
        end_transform = end_waypoint.transform
        end_transform.location.z += 0.5
        ego_vehicle = CarlaDataProvider.request_new_actor('vehicle.tesla.model3', end_transform)
        self.other_actors.append(ego_vehicle)

    def _create_behavior(self):
        """
        The scenario defined after is a "follow leading vehicle" scenario. After
        invoking this scenario, it will wait for the user controlled vehicle to
        enter the start region, then make the other actor to drive until reaching
        the next intersection. Finally, the user-controlled vehicle has to be close
        enough to the other actor to end the scenario.
        If this does not happen within 60 seconds, a timeout stops the scenario
        """

        # Ego vehicle must drive unto end point 
        end_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._distance)
        end_transform = end_waypoint.transform
        end_transform.location.z += 0.5

        # end condition
        endcondition = py_trees.composites.Parallel("Waiting for end position",
                                                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
        endcondition_part1 = InTriggerDistanceToLocation(actor=self.ego_vehicles[0],
                                                         target_location=end_transform.location,
                                                         distance=10,
                                                         name="FinalDistanceToEndPoint")
        endcondition.add_child(endcondition_part1)

        # Build behavior tree
        sequence = py_trees.composites.Sequence("Sequence Behavior")
        sequence.add_child(endcondition)

        return sequence


    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        collision_criterion = CollisionTest(self.ego_vehicles[0])

        criteria.append(collision_criterion)

        return criteria
    
    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()