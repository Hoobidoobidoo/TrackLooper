#!/bin/env python

import json
from tqdm import tqdm
from Module import Module
import math

ptthresh = 0.8

# #                       # {
# #       detId           #     "411309061": [
# #       boundary 1      #         [
# #          z coord      #             -129.13503188311316,
# #          x coord      #             27.99223639171157,
# #          y coord      #             -0.7045785443457637
# #                       #         ],
# #       boundary 2      #         [
# #          z coord      #             -129.13503188311316,
# #          x coord      #             26.454739008300713,
# #          y coord      #             9.176519663647351
# #                       #         ],
# #       boundary 3      #         [
# #          z coord      #             -129.13503188311316,
# #          x coord      #             21.514189904304153,
# #          y coord      #             8.407770971941925
# #                       #         ],
# #       boundary 4      #         [
# #          z coord      #             -129.13503188311316,
# #          x coord      #             23.051687287715012,
# #          y coord      #             -1.4733272360511913
# #                       #         ]
# #                       #     ],
# #                       #     "411309062": [
# #                       #         [
# #                       #             -128.7349884581733,
# #                       #             28.00071896416048,
# #                       #             -0.693051092358993
# #                       #         ],
# #                       #         [
# #                       #             -128.7349884581733,
# #                       #             26.4596055174114,
# #                       #             9.187483779005468
# #                       #         ],
# #                       #         [
# #                       #             -128.7349884581733,
# #                       #             21.51933808172917,
# #                       #             8.416927055630929
# #                       #         ],
# #                       #         [
# #                       #             -128.7349884581733,
# #                       #             23.06045152847825,
# #                       #             -1.4636078157335337
# #                       #         ]
# #                       #     ],
# #                       #     ...
# #                       #     ...
# #                       #     ...
# #                       #     ...
# #                       #     ...
# #                       # }

k = 0.00299792458
B = 3.8
A = k * B / 2.

def Phi_mpi_pi(phi):
    while phi >= math.pi: phi -= 2. * math.pi;
    while phi < -math.pi: phi += 2. * math.pi;
    return phi;

class DetectorGeometry:

    def __init__(self, data):
        self.data = data
        self.datafile = open(self.data)
        self.geom_data_raw = json.load(self.datafile)
        self.geom_data = {}
        for key in tqdm(self.geom_data_raw, "Loading detector geometries (i.e. boundaries)"):
            detid = int(key)
            new_list = []
            for x in self.geom_data_raw[key]:
                new_x = [ float(y) for y in x ]
                new_list.append(new_x)
            self.geom_data[detid] = new_list

    def getData(self, filt=None):
        if filt:
            rtndict = dict(filter(filt, self.geom_data.items()))
            return rtndict
        else:
            return self.geom_data

    def getDetIds(self, filt=None):
        if filt:
            rtndict = dict(filter(filt, self.geom_data.items()))
            return rtndict.keys()
        else:
            return self.geom_data.keys()

    def buildByLayer(self):
        self.barrel_lower_det_ids = []
        print("Building barrel detIds")
        for i in tqdm(range(1, 7)):
            self.barrel_lower_det_ids.append(
                    self.getDetIds(lambda x:
                            Module(x[0]).subdet() == 5 and
                            Module(x[0]).layer() == i and
                            Module(x[0]).isLower() == 1
                            )
                    )
        self.endcap_lower_det_ids = []
        print("Building endcap detIds")
        for i in tqdm(range(1, 6)):
            self.endcap_lower_det_ids.append(
                    self.getDetIds(lambda x:
                            Module(x[0]).subdet() == 4 and
                            Module(x[0]).layer() == i and
                            Module(x[0]).isLower() == 1
                            )
                    )

    def getBarrelLayerDetIds(self, layer):
        return self.barrel_lower_det_ids[layer-1]

    def getEndcapLayerDetIds(self, layer):
        return self.endcap_lower_det_ids[layer-1]

    def getMinR(self, detid):
        points = self.geom_data[detid]
        rs = []
        for point in points:
            rs.append(math.sqrt(point[1]**2 + point[2]**2))
        return min(rs)

    def getMaxR(self, detid):
        points = self.geom_data[detid]
        rs = []
        for point in points:
            rs.append(math.sqrt(point[1]**2 + point[2]**2))
        return max(rs)

    def getMinPhi(self, detid):
        points = self.geom_data[detid]
        phis = []
        posphis = []
        negphis = []
        signs = []
        bigger_than_pi_over_2 = []
        for point in points:
            phi = Phi_mpi_pi(math.pi + math.atan2(-point[2],-point[1]))
            phis.append(phi)
            if phi > 0:
                posphis.append(phi)
            else:
                negphis.append(phi)
            signs.append(phi > 0)
            bigger_than_pi_over_2.append(abs(phi) > math.pi / 2.)
        if sum(signs) == 4 or sum(signs) == 0:
            return min(phis)
        elif sum(bigger_than_pi_over_2) == 4:
            return min(posphis)
        else:
            return min(phis)

    def getMaxPhi(self, detid):
        points = self.geom_data[detid]
        phis = []
        posphis = []
        negphis = []
        signs = []
        bigger_than_pi_over_2 = []
        for point in points:
            phi = Phi_mpi_pi(math.pi + math.atan2(-point[2],-point[1]))
            phis.append(phi)
            if phi > 0:
                posphis.append(phi)
            else:
                negphis.append(phi)
            signs.append(phi > 0)
            bigger_than_pi_over_2.append(abs(phi) > math.pi / 2.)
        if sum(signs) == 4 or sum(signs) == 0:
            return max(phis)
        elif sum(bigger_than_pi_over_2) == 4:
            return max(negphis)
        else:
            return max(phis)

    def getMinZ(self, detid):
        points = self.geom_data[detid]
        zs = []
        for point in points:
            zs.append(point[0])
        return min(zs)

    def getMaxZ(self, detid):
        points = self.geom_data[detid]
        zs = []
        for point in points:
            zs.append(point[0])
        return max(zs)

    def getCompatiblePhiRange(self, detid, ptmin, ptmax):
        minr = self.getMinR(detid)
        maxr = self.getMaxR(detid)
        minphi = self.getMinPhi(detid)
        maxphi = self.getMaxPhi(detid)
        pos_q_phi_lo_bound = Phi_mpi_pi(A * minr / ptmax + minphi)
        pos_q_phi_hi_bound = Phi_mpi_pi(A * maxr / ptmin + maxphi)
        neg_q_phi_lo_bound = Phi_mpi_pi(-A * maxr / ptmin + minphi)
        neg_q_phi_hi_bound = Phi_mpi_pi(-A * minr / ptmax + maxphi)
        return [[pos_q_phi_lo_bound, pos_q_phi_hi_bound], [neg_q_phi_lo_bound, neg_q_phi_hi_bound]]

    def getCompatibleEtaRange(self, detid, zmin_bound, zmax_bound):
        minr = self.getMinR(detid)
        maxr = self.getMaxR(detid)
        minz = self.getMinZ(detid)
        maxz = self.getMaxZ(detid)
        if minz > 0:
            maxeta = -math.log(math.tan(math.atan2(maxr, (minz - zmin_bound)) / 2. ))
        else:
            maxeta = -math.log(math.tan(math.atan2(minr, (minz - zmin_bound)) / 2. ))
        if maxz > 0:
            mineta = -math.log(math.tan(math.atan2(minr, (maxz - zmax_bound)) / 2. ))
        else:
            mineta = -math.log(math.tan(math.atan2(maxr, (maxz - zmax_bound)) / 2. ))
        return sorted([mineta, maxeta], key=lambda eta: eta)

    def isConnected(self, detid, etamin, etamax, phimin, phimax, ptmin, ptmax, zmin=-30, zmax=30, verbose=False):

        # Check Phi
        phirange = self.getCompatiblePhiRange(detid, ptmin, ptmax)
        if verbose:
            print(phimin, phimax, phirange)
        if verbose:
            print(Phi_mpi_pi(phimin - phirange[0][0]))
            print(Phi_mpi_pi(phimin - phirange[0][1]))
            print(Phi_mpi_pi(phimax - phirange[0][0]))
            print(Phi_mpi_pi(phimax - phirange[0][1]))
        data = []
        if abs(Phi_mpi_pi(phimin - phirange[0][0])) < math.pi/2.: data.append(Phi_mpi_pi(phimin - phirange[0][0]) > 0)
        if abs(Phi_mpi_pi(phimin - phirange[0][1])) < math.pi/2.: data.append(Phi_mpi_pi(phimin - phirange[0][1]) > 0)
        if abs(Phi_mpi_pi(phimax - phirange[0][0])) < math.pi/2.: data.append(Phi_mpi_pi(phimax - phirange[0][0]) > 0)
        if abs(Phi_mpi_pi(phimax - phirange[0][1])) < math.pi/2.: data.append(Phi_mpi_pi(phimax - phirange[0][1]) > 0)
        if len(data) != 4:
            return False;
        if all(data) or not any(data):
            is_phi_in_range_0 = False
        else:
            is_phi_in_range_0 = True

        if verbose:
            print(data)
            print(all(data))
            print(any(data))
        data = []
        if abs(Phi_mpi_pi(phimin - phirange[1][0])) < math.pi/2.: data.append(Phi_mpi_pi(phimin - phirange[1][0]) > 0)
        if abs(Phi_mpi_pi(phimin - phirange[1][1])) < math.pi/2.: data.append(Phi_mpi_pi(phimin - phirange[1][1]) > 0)
        if abs(Phi_mpi_pi(phimax - phirange[1][0])) < math.pi/2.: data.append(Phi_mpi_pi(phimax - phirange[1][0]) > 0)
        if abs(Phi_mpi_pi(phimax - phirange[1][1])) < math.pi/2.: data.append(Phi_mpi_pi(phimax - phirange[1][1]) > 0)
        if len(data) != 4:
            return False;
        if all(data) or not any(data):
            is_phi_in_range_1 = False
        else:
            is_phi_in_range_1 = True

        if verbose:
            print(data)
            print(all(data))
            print(any(data))

        if verbose:
            print(is_phi_in_range_0)
            print(is_phi_in_range_1)

        if not is_phi_in_range_0 and not is_phi_in_range_1:
            return False

        # Check Eta
        etarange = self.getCompatibleEtaRange(detid, zmin, zmax)
        if verbose:
            print(etamin, etamax, etarange)
        data = []
        data.append((etamin - etarange[0]) > 0)
        data.append((etamin - etarange[1]) > 0)
        data.append((etamax - etarange[0]) > 0)
        data.append((etamax - etarange[1]) > 0)
        if all(data) or not any(data):
            is_eta_in_range = False
        else:
            is_eta_in_range = True
        if not is_eta_in_range:
            return False

        return True
