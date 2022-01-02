import numpy as np
from simple_playgrounds.common.position_utils import CoordinateSampler, Trajectory
from simple_playgrounds.common.spawner import Spawner
from simple_playgrounds.element.elements.aura import Fireball
from simple_playgrounds.element.elements.basic import Physical
from simple_playgrounds.element.elements.contact import Candy, Poison
from simple_playgrounds.playground.playground import Playground, PlaygroundRegister


class PlainPG(Playground):
    def __init__(self, time_limit=100, size=(200, 200)):
        super().__init__()

        self.time_limit = time_limit
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


@PlaygroundRegister.register("custom", "candy_poison")
class CandyPoison(PlainPG):
    def __init__(self, time_limit=100, size=(200, 200), probability_production=0.4):
        super().__init__(time_limit, size)

        # Foraging
        area_prod = CoordinateSampler(
            center=self._center, area_shape="rectangle", size=self._size
        )

        spawner = Spawner(
            Candy, production_area=area_prod, probability=probability_production
        )
        self.add_spawner(spawner)

        spawner = Spawner(
            Poison, production_area=area_prod, probability=probability_production
        )
        self.add_spawner(spawner)


@PlaygroundRegister.register("custom", "candy_fireballs")
class CandyFireballs(PlainPG):
    def __init__(self, time_limit=100, size=(200, 200), probability_production=0.4):
        super().__init__(time_limit, size)
        coord_scaler = max(size) / 200

        fireball_texture = {"texture_type": "centered_random_tiles", "size_tiles": 4}
        interaction_range = 10 * coord_scaler

        # First Fireball
        text = {"color_min": [220, 0, 200], "color_max": [255, 100, 220]}
        waypoints = (
            np.array([[20, 20], [20, 180], [180, 180], [180, 20]]) * coord_scaler
        )
        trajectory = Trajectory(
            "waypoints",
            trajectory_duration=300,
            waypoints=waypoints,
        )
        fireball = Fireball(
            reward=-1,
            texture={**fireball_texture, **text},
            invisible_range=interaction_range,
        )
        self.add_element(fireball, trajectory)

        # Second Fireball
        text = {"color_min": [180, 0, 0], "color_max": [220, 100, 0]}
        waypoints = np.array([[40, 40], [160, 160]]) * coord_scaler
        trajectory = Trajectory(
            "waypoints",
            trajectory_duration=150,
            waypoints=waypoints,
        )
        fireball = Fireball(
            reward=-2,
            texture={**fireball_texture, **text},
            invisible_range=interaction_range,
        )
        self.add_element(fireball, trajectory)

        # Third Fireball
        text = {"color_min": [220, 100, 0], "color_max": [255, 120, 0]}
        waypoints = np.array([[40, 160], [160, 40]]) * coord_scaler
        trajectory = Trajectory(
            "waypoints",
            trajectory_duration=180,
            waypoints=waypoints,
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
            Candy, production_area=area_prod, probability=probability_production
        )
        self.add_spawner(spawner)


@PlaygroundRegister.register("custom", "candy_poison_large")
class CandyPoisonLarge(CandyPoison):
    def __init__(self, time_limit=100, size=(1000, 1000), probability_production=0.4):
        super().__init__(time_limit, size)


@PlaygroundRegister.register("custom", "candy_fireballs_large")
class CandyFireballsLarge(CandyFireballs):
    def __init__(self, time_limit=100, size=(1000, 1000), probability_production=0.4):
        super().__init__(time_limit, size)
