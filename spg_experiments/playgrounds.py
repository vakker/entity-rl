from simple_playgrounds.common.position_utils import CoordinateSampler, Trajectory
from simple_playgrounds.common.spawner import Spawner
from simple_playgrounds.element.elements.aura import Fireball
from simple_playgrounds.element.elements.basic import Physical
from simple_playgrounds.element.elements.contact import Candy, Poison
from simple_playgrounds.playground.playground import Playground, PlaygroundRegister


class PlainPG(Playground):
    def __init__(self, time_limit=100):
        super().__init__()

        self.time_limit = time_limit
        self._size = (200, 200)
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
class CandyPosion(PlainPG):
    def __init__(self, time_limit=100, probability_production=0.4):
        super().__init__(time_limit)

        # Foraging
        area_prod = CoordinateSampler(
            center=(100, 100), area_shape="rectangle", size=(150, 150)
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
    def __init__(self, time_limit=100, probability_production=0.4):
        super().__init__(time_limit)

        fireball_texture = {"texture_type": "centered_random_tiles", "size_tiles": 4}

        # First Fireball
        text_1 = {"color_min": [220, 0, 200], "color_max": [255, 100, 220]}
        trajectory = Trajectory(
            "waypoints",
            trajectory_duration=300,
            waypoints=[[20, 20], [20, 180], [180, 180], [180, 20]],
        )
        fireball = Fireball(reward=-1, texture={**fireball_texture, **text_1})
        self.add_element(fireball, trajectory)

        # Second Fireball
        text_2 = {"color_min": [180, 0, 0], "color_max": [220, 100, 0]}
        trajectory = Trajectory(
            "waypoints", trajectory_duration=150, waypoints=[[40, 40], [160, 160]]
        )
        fireball = Fireball(reward=-2, texture={**fireball_texture, **text_2})
        self.add_element(fireball, trajectory)

        # Third Fireball
        text_3 = {"color_min": [220, 100, 0], "color_max": [255, 120, 0]}
        trajectory = Trajectory(
            "waypoints", trajectory_duration=180, waypoints=[[40, 160], [160, 40]]
        )
        fireball = Fireball(reward=-5, texture={**fireball_texture, **text_3})
        self.add_element(fireball, trajectory)

        # Foraging
        area_prod = CoordinateSampler(
            center=(100, 100), area_shape="rectangle", size=(150, 150)
        )

        spawner = Spawner(
            Candy, production_area=area_prod, probability=probability_production
        )
        self.add_spawner(spawner)
