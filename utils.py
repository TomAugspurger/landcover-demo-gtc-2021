from numba import stencil, njit
import matplotlib.colors
import numpy as np

NUM_CLASSES = 13

@stencil(neighborhood=((-3, 2), (-3, 2)))
def _smooth(x):
    base = result = x[0, 0]
    for i in range(-3, 3):
        for j in range(-3, 3):
            if x[i, j] != base:
                result = 0
                break
    return result


@njit(parallel=True)
def smooth(x):
    return _smooth(x)


# -------------
# Hack to serialize stencil functions.
# Probably breaks every other numba thing out there, so don't use this.
# See https://github.com/numba/numba/pull/6660

def reconstruct(is_smooth):
    assert is_smooth
    return smooth


def reduce(obj):
    return reconstruct, (obj.__name__ == "smooth",)

import copyreg
copyreg.pickle(type(smooth), reduce, reconstruct)
# -----------------

   
chesapeake_class_definitions = {
    0: ("No Data", "Background Values"),
    1: ("Water", "All areas of open water. This includes ponds, rivers, lakes and boats not attached to docks. MMU = 25 square meters"),
    2: ("Emergent Wetlands", "Low vegetation areas located along marine or estuarine regions that are visually confirmed to have the look of saturated ground surrounding the vegetation and that are located along major waterways (i.e. rivers, ocean). MMU = 225 square meters"),
    3: ("Tree Canopy", "Deciduous and evergreen woody vegetation of either natural succession or human planting that is over approximately 3-5 meters in height. Stand-alone individuals, discrete clumps, and interlocking individuals are included. MMU = 9 square meters"),
    4: ("Shrubland", "Heterogeneous area of both/either deciduous and/or evergreen woody vegetation. Characterized by variation in height of vegetation through patchy coverage of shrubs and young trees interspersed with grasses and other lower vegetation. Discrete clumps and small patches of interlocking individuals are included, as are true shrubs, young trees, and trees or shrubs that are small or stunted because of environmental conditions, when intermingled in a heterogeneous landscape with low vegetation. MMU = 225 square meters"),
    5: ("Low Vegetation", "Plant material less than approximately 2 meters in height. Includes lawns, tilled fields, nursery plantings with or without tarp cover and natural ground cover. MMU = 9 square meters"),
    6: ("Barren", "Areas void of vegetation consisting of natural earthen material regardless of how it has been cleared. This includes beaches, mud flats, and bare ground in construction sites. MMU = 25 square meters"),
    7: ("Structures", "Human-constructed objects made of impervious materials that are greater than approximately 2 meters in height. Houses, malls, and electrical towers are examples of structures. MMU = 9 square meters"),
    8: ("Impervious Surfaces", "Human-constructed surfaces through which water cannot penetrate, and that are below approximately 2 meters in height. MMU = 9 square meters"),
    9: ("Impervious Roads", "Impervious surfaces that are used and maintained for transportation. MMU = 9 square meters"),
    10: ("Tree Canopy over Structures", "Forest or Tree Cover that overlaps with impervious surfaces rendering the structures partially or completely not visible to plain sight. MMU = 9 square meters"),
    11: ("Tree Canopy over Impervious Surfaces", "Forest or Tree Cover that overlaps with impervious surfaces rendering the impervious surface partially or completely not visible to plain sight. MMU = 9 square meters"),
    12: ("Tree Canopy over Impervious Roads", "Forest or Tree Cover that overlaps with impervious surfaces rendering the roads partially or completely not visible to plain sight. MMU = 9 square meters"),
    13: ("Aberdeen Proving Ground", "Should be ignored as a background value")
}


lc_colormap = {
    0: (0, 0, 0, 0),
    1: (0, 197, 255, 255),
    2: (0, 168, 132, 255),
    3: (38, 115, 0, 255),
    4: (76, 230, 0, 255),
    5: (163, 255, 115, 255),
    6: (255, 170, 0, 255),
    7: (255, 0, 0, 255),
    8: (156, 156, 156, 255),
    9: (0, 0, 0, 255),
    10: (115, 115, 0, 255),
    11: (230, 230, 0, 255),
    12: (255, 255, 115, 255),
    13: (197, 0, 255, 255)
}
temp_colors = []
for idx in range(0, 14):
    r,g,b,a = lc_colormap[idx]
    temp_colors.append([r,g,b])
temp_colors = np.array(temp_colors) / 255.0

lc_cmap = matplotlib.colors.ListedColormap(temp_colors)