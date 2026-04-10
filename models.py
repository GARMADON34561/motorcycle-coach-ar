"""Motorcycle AR Coach - Action and Observation models."""

from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State


class MotorcycleAction(Action):
    throttle: float = Field(ge=0.0, le=1.0, description="Throttle 0-1")
    brake: float = Field(ge=0.0, le=1.0, description="Brake 0-1")
    lean_angle: float = Field(ge=-45.0, le=45.0, description="Lean angle in degrees")
    steering: float = Field(ge=-1.0, le=1.0, description="Steering -1 left to 1 right")


class MotorcycleObservation(Observation):
    speed_kmh: float = Field(description="Current speed in km/h")
    lean_angle: float = Field(description="Current lean angle in degrees")
    distance_to_obstacle_m: float = Field(description="Distance to nearest obstacle in meters")
    fuel_level_l: float = Field(description="Fuel level in liters")
    road_condition: str = Field(description="dry, wet, gravel, or ice")
    turn_radius_m: float = Field(description="Current turn radius in meters")
    headway_seconds: float = Field(description="Time gap to vehicle ahead in seconds")
    done: bool = Field(default=False)
    reward: float = Field(default=0.0)


class MotorcycleState(State):
    current_task_index: int = Field(default=0)
    total_reward: float = Field(default=0.0)
    step_count: int = Field(default=0)
