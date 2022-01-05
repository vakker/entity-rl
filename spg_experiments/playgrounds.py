# pylint: disable=no-member,too-few-public-methods


import numpy as np
from simple_playgrounds.common.definitions import CollisionTypes
from simple_playgrounds.common.position_utils import CoordinateSampler, Trajectory
from simple_playgrounds.common.spawner import Spawner
from simple_playgrounds.element.elements.activable import Dispenser
from simple_playgrounds.element.elements.aura import Fireball
from simple_playgrounds.element.elements.basic import Physical
from simple_playgrounds.element.elements.contact import Candy, Poison
from simple_playgrounds.element.elements.teleport import Portal as SPGPortal
from simple_playgrounds.element.elements.teleport import PortalColor
from simple_playgrounds.playground.layouts import SingleRoom as SPGSingleRoom
from simple_playgrounds.playground.playground import Playground, PlaygroundRegister


class TouchDispenser(Dispenser):
    def _set_shape_collision(self):
        self.pm_invisible_shape.collision_type = CollisionTypes.CONTACT


class Portal(SPGPortal):
    def __init__(self, color):
        super().__init__(color)
        self.energized = False

    def energize(self, agent):
        super().energize(agent)
        self.energized = True

    def pre_step(self):
        super().pre_step()
        self.energized = False


class Pole(Physical):
    def __init__(self):
        super().__init__(config_key="circle", radius=2)


class PlainPG(Playground):
    def __init__(self, size=(200, 200)):
        super().__init__()

        self._size = size
        self._width, self._length = self._size
        self._center = (self._width / 2, self._length / 2)

        pole = Pole()
        self.add_element(pole, [self._center, 0])

        self.initial_agent_coordinates = CoordinateSampler(
            center=self._center,
            area_shape="rectangle",
            size=self._size,
        )


class SingleRoom(SPGSingleRoom):
    def __init__(self, *args, **kwargs):
        # Same at "texture_type": "color", but all this is needed
        # because this is what the texture generator expects.
        kwargs["wall_texture"] = {
            "texture_type": "random_tiles",
            "color_min": [150, 150, 0],
            "color_max": [150, 150, 0],
            "size_tiles": 4,
        }
        super().__init__(*args, **kwargs)


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
            production_limit=400,
        )
        self.add_spawner(spawner)


class DispenserFireballsBase:
    def __init__(self, size=(200, 200)):
        super().__init__(size)

        coord_scaler = max(size) / 200

        fireball_texture = {"texture_type": "color"}
        interaction_range = 10 * coord_scaler
        waypoints = (np.array([[100, 20], [100, 180]]) * coord_scaler).tolist()

        # First Fireball
        text = {"color": [235, 50, 210]}
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
        text = {"color": [200, 50, 0]}
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
        text = {"color": [235, 110, 0]}
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

        area_prod = CoordinateSampler(
            center=[self._center[0] / 2, self._center[1]],
            area_shape="rectangle",
            size=[s / 3 for s in self._size],
        )

        portal_red = Portal(color=PortalColor.RED)
        self.add_element(portal_red, ([7, 0.5 * self._size[1]], np.pi))
        portal_blue = Portal(color=PortalColor.BLUE)
        self.add_element(portal_blue, ([self._size[0] - 7, 0.5 * self._size[1]], 0))
        self.portals = [portal_red, portal_blue]

        portal_red.destination = portal_blue
        portal_blue.destination = portal_red

        disp_coord = [0.9 * self._size[0], 0.9 * self._size[1]]
        dispenser = TouchDispenser(
            element_produced=Candy,
            production_area=area_prod,
            production_limit=10 * coord_scaler,
            radius=10,
            allow_overlapping=False,
        )
        self.add_element(dispenser, [disp_coord, 0])


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


@PlaygroundRegister.register("nowall", "dispenser_fireballs")
class NoWallDispenserFireballs(DispenserFireballsBase, PlainPG):
    pass


@PlaygroundRegister.register("wall", "dispenser_fireballs")
class WallDispenserFireballs(DispenserFireballsBase, SingleRoom):
    pass


@PlaygroundRegister.register("nowall", "dispenser_fireballs_large")
class NoWallDispenserFireballsLarge(NoWallDispenserFireballs):
    def __init__(self, size=(1000, 1000)):
        super().__init__(size)


@PlaygroundRegister.register("wall", "dispenser_fireballs_large")
class WallDispenserFireballsLarge(WallDispenserFireballs):
    def __init__(self, size=(1000, 1000)):
        super().__init__(size)
