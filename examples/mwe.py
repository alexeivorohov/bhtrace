from bhtrace.geometry import MinkowskiCart, Photon
from bhtrace.grrt import Scene, ThinNewtonianDisk


ST = MinkowskiCart()

disk = ThinNewtonianDisk(ST)

photon = Photon(ST)

scene = Scene(ST, Photon, net, disk)

IM1 = scene.compute(angle= ,saveas='', method=ptracer)

IM2 = scene.compute(angle=,saveas='',)