"""Motorcycle AR Coach - Core environment logic."""

from uuid import uuid4
from typing import Optional
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import MotorcycleAction, MotorcycleObservation, MotorcycleState


class MotorcycleEnvironment(Environment[MotorcycleAction, MotorcycleObservation, MotorcycleState]):
    def __init__(self):
        self._state = MotorcycleState(
            episode_id=str(uuid4()),
            step_count=0,
            current_task_index=0,
            total_reward=0.0
        )
        self.tasks = [
            {"type": "corner", "initial_speed": 40.0, "turn_radius": 15.0, "road_condition": "dry", "max_safe_lean": 35.0},
            {"type": "emergency", "initial_speed": 60.0, "obstacle_distance": 20.0, "road_condition": "wet"},
            {"type": "cruise", "distance_km": 5.0, "traffic_density": 0.5, "road_condition": "dry"}
        ]
    
    def reset(self, seed=None, episode_id=None, **kwargs):
        self._state = MotorcycleState(episode_id=episode_id or str(uuid4()), step_count=0, current_task_index=0, total_reward=0.0)
        return self._get_initial_observation()
    
    def _get_initial_observation(self):
        task = self.tasks[self._state.current_task_index]
        if task["type"] == "corner":
            return MotorcycleObservation(speed_kmh=task["initial_speed"], lean_angle=0.0, distance_to_obstacle_m=999.0, fuel_level_l=5.0, road_condition=task["road_condition"], turn_radius_m=task["turn_radius"], headway_seconds=999.0, done=False, reward=0.0)
        elif task["type"] == "emergency":
            return MotorcycleObservation(speed_kmh=task["initial_speed"], lean_angle=0.0, distance_to_obstacle_m=task["obstacle_distance"], fuel_level_l=5.0, road_condition=task["road_condition"], turn_radius_m=999.0, headway_seconds=999.0, done=False, reward=0.0)
        else:
            return MotorcycleObservation(speed_kmh=50.0, lean_angle=0.0, distance_to_obstacle_m=999.0, fuel_level_l=10.0, road_condition=task["road_condition"], turn_radius_m=999.0, headway_seconds=2.0, done=False, reward=0.0)
    
    def step(self, action, **kwargs):
        task = self.tasks[self._state.current_task_index]
        reward = 0.0
        if task["type"] == "corner":
            ideal_lean = min(35.0, (action.throttle * 40)**2 / (task["turn_radius"] * 9.8) * 10)
            lean_error = abs(action.lean_angle - ideal_lean) / task["max_safe_lean"]
            raw = 1.0 - min(1.0, lean_error)
            reward = max(0.01, min(0.99, raw))
        elif task["type"] == "emergency":
            stopping_distance = (action.brake * 15) + (abs(action.steering) * 5)
            if stopping_distance >= task["obstacle_distance"]:
                reward = 0.01
            else:
                smoothness = 1.0 - abs(action.steering) * 0.5
                raw = 0.5 + smoothness * 0.4
                reward = max(0.01, min(0.99, raw))
        else:
            fuel_used = action.throttle * 0.5
            headway_safety = min(1.0, max(0.0, (action.brake * 2 + 1) / 3))
            raw = (1 - fuel_used/5) * 0.5 + headway_safety * 0.5
            reward = max(0.01, min(0.99, raw))
        
        self._state.current_task_index += 1
        self._state.step_count += 1
        self._state.total_reward += reward
        done = self._state.current_task_index >= len(self.tasks)
        
        if done:
            final_score = self._state.total_reward / len(self.tasks)
            final_score = max(0.01, min(0.99, final_score))
            obs = MotorcycleObservation(speed_kmh=0, lean_angle=0, distance_to_obstacle_m=0, fuel_level_l=0, road_condition="", turn_radius_m=0, headway_seconds=0, done=True, reward=final_score)
        else:
            obs = self._get_initial_observation()
            obs.reward = reward
            obs.done = False
        return obs
    
    @property
    def state(self):
        return self._state
