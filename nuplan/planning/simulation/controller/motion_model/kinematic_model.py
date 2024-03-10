import numpy as np

from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
from nuplan.common.actor_state.ego_state import EgoState, EgoStateDot
from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters, get_pacifica_parameters
from nuplan.planning.simulation.controller.motion_model.abstract_motion_model import AbstractMotionModel

class BasicKinematicModel(AbstractMotionModel):
    def __init__(
            self,
            vehicle: VehicleParameters,
            max_steering_angle: float = np.pi / 3,
            accel_time_constant: float = 0.2,
            steering_angle_time_constant: float = 0.05,
            model_type = "cv"
    ):
        """
        Construct KinematicModel.

        :param vehicle: Vehicle parameters.
        :param max_steering_angle: [rad] Maximum absolute value steering angle allowed by model.
        :param accel_time_constant: low pass filter time constant for acceleration in s
        :param steering_angle_time_constant: low pass filter time constant for steering angle in s
        """
        self.vehicle = vehicle
        self.max_steering_angle = max_steering_angle
        self.accel_time_constant = accel_time_constant
        self.steering_angle_time_constant = steering_angle_time_constant
        self.model_type = model_type
        self._vehicle = vehicle

    def get_state_dot(self, state: EgoState) -> EgoStateDot:
        pass

    def _cv_model(self, state: EgoState, sampling_time: TimePoint) -> EgoState:
        return state
    
    def _ca_model(self, state: EgoState, sampling_time: TimePoint) -> EgoState:
        return state
    
    def _ckappa_model(self, state: EgoState, sampling_time: TimePoint) -> EgoState:
        return state
    
    def propagate_state(self, state: EgoState, ideal_dynamic_state: DynamicCarState, sampling_time: TimePoint) -> EgoState:
        propagating_state = state
        if self.model_type == "cv":
            propagating_state = self._cv_model(propagating_state, sampling_time)
        elif self.model_type == "ca":
            propagating_state = self._ca_model(propagating_state, sampling_time)
        elif self.model_type == "ckappa":
            propagating_state = self._ckappa_model(propagating_state, sampling_time)


if __name__ == '__main__':
    vehicle = get_pacifica_parameters()
    basic_kinematic_model = BasicKinematicModel(vehicle)