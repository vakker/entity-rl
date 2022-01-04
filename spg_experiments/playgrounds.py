# pylint: disable=no-member,too-few-public-methods


import numpy as np
from simple_playgrounds.common.definitions import CollisionTypes, ElementTypes
from simple_playgrounds.common.position_utils import CoordinateSampler, Trajectory
from simple_playgrounds.common.spawner import Spawner
from simple_playgrounds.element.element import InteractiveElement, SceneElement
from simple_playgrounds.element.elements.activable import ActivableElement, Dispenser
from simple_playgrounds.element.elements.aura import Fireball
from simple_playgrounds.element.elements.basic import Physical
from simple_playgrounds.element.elements.contact import Candy, Poison
from simple_playgrounds.element.elements.teleport import (
    InvisibleBeam,
    Portal,
    PortalColor,
    VisibleBeamHoming,
)
from simple_playgrounds.playground.layouts import SingleRoom
from simple_playgrounds.playground.playground import Playground, PlaygroundRegister


class TouchDispenser(Dispenser):
    def _set_shape_collision(self):
        self.pm_invisible_shape.collision_type = CollisionTypes.CONTACT


class PlainPG(Playground):
    def __init__(self, size=(200, 200)):
        super().__init__()

        self._size = size
        self._width, self._length = self._size
        self._center = (self._width / 2, self._length / 2)

        pole = Physical(config_key="circle", radius=2)
        self.add_element(pole, [self._center, 0])

        self.initial_agent_coordinates = CoordinateSampler(
            center=self._center,
            area_shape="rectangle",
            size=self._size,
        )


class CandyPoisonBase:
    def __init__(self, size=(200, 200), probability_production=0.4):
        super().__init__(size)
        coord_scaler = max(size) / 200

        # Foraging
        area_prod = CoordinateSampler(
            center=self._center, area_shape="rectangle", size=self._size
        )

        spawner = Spawner(
            Candy,
            production_area=area_prod,
            probability=probability_production,
            max_elements_in_playground=10 * coord_scaler,
            production_limit=200,
        )
        self.add_spawner(spawner)

        spawner = Spawner(
            Poison,
            production_area=area_prod,
            probability=probability_production,
            max_elements_in_playground=10 * coord_scaler,
            production_limit=200,
        )
        self.add_spawner(spawner)


class CandyFireballsBase:
    def __init__(self, size=(200, 200), probability_production=0.4):
        super().__init__(size)
        coord_scaler = max(size) / 200

        fireball_texture = {"texture_type": "color"}
        interaction_range = 10 * coord_scaler

        # First Fireball
        text = {"color": [235, 50, 210]}
        waypoints = (
            np.array([[20, 20], [20, 180], [180, 180], [180, 20]]) * coord_scaler
        )

        trajectory = Trajectory(
            "waypoints",
            trajectory_duration=300,
            waypoints=waypoints.tolist(),
        )
        fireball = Fireball(
            reward=-1,
            texture={**fireball_texture, **text},
            invisible_range=interaction_range,
        )
        self.add_element(fireball, trajectory)

        # Second Fireball
        text = {"color": [200, 50, 0]}
        waypoints = np.array([[40, 40], [160, 160]]) * coord_scaler
        trajectory = Trajectory(
            "waypoints",
            trajectory_duration=150,
            waypoints=waypoints.tolist(),
        )
        fireball = Fireball(
            reward=-2,
            texture={**fireball_texture, **text},
            invisible_range=interaction_range,
        )
        self.add_element(fireball, trajectory)

        # Third Fireball
        text = {"color": [235, 110, 0]}
        waypoints = np.array([[40, 160], [160, 40]]) * coord_scaler
        trajectory = Trajectory(
            "waypoints",
            trajectory_duration=180,
            waypoints=waypoints.tolist(),
        )
        fireball = Fireball(
            reward=-5,
            texture={**fireball_texture, **text},
            invisible_range=interaction_range,
        )
        self.add_element(fireball, trajectory)

        # Foraging
        area_prod = CoordinateSampler(
            center=self._center, area_shape="rectangle", size=self._size
        )

        spawner = Spawner(
            Candy,
            production_area=area_prod,
            probability=probability_production,
            max_elements_in_playground=20 * coord_scaler,
            production_limit=200,
        )
        self.add_spawner(spawner)


@PlaygroundRegister.register("nowall", "candy_poison")
class NoWallCandyPoison(CandyPoisonBase, PlainPG):
    pass


@PlaygroundRegister.register("nowall", "candy_fireballs")
class NoWallCandyFireballs(CandyFireballsBase, PlainPG):
    pass


@PlaygroundRegister.register("nowall", "candy_poison_large")
class NoWallCandyPoisonLarge(NoWallCandyPoison):
    def __init__(self, size=(1000, 1000), probability_production=0.4):
        super().__init__(size, probability_production=0.4)


@PlaygroundRegister.register("nowall", "candy_fireballs_large")
class NoWallCandyFireballsLarge(NoWallCandyFireballs):
    def __init__(self, size=(1000, 1000), probability_production=0.4):
        super().__init__(size, probability_production=0.4)


@PlaygroundRegister.register("wall", "candy_poison")
class WallCandyPoison(CandyPoisonBase, SingleRoom):
    pass


@PlaygroundRegister.register("wall", "candy_fireballs")
class WallCandyFireballs(CandyFireballsBase, SingleRoom):
    pass


@PlaygroundRegister.register("wall", "candy_poison_large")
class WallCandyPoisonLarge(WallCandyPoison):
    def __init__(self, size=(1000, 1000), probability_production=0.4):
        super().__init__(size, probability_production=0.4)


@PlaygroundRegister.register("wall", "candy_fireballs_large")
class WallCandyFireballsLarge(WallCandyFireballs):
    def __init__(self, size=(1000, 1000), probability_production=0.4):
        super().__init__(size, probability_production=0.4)
