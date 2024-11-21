#!/bin/env python3

# Things that may need installation
# pip3 install plotly==5.20.0
# numpy
# tqdm

import sys
import plotly
import plotly.graph_objs as go
import plotly.express as px
import LSTMath as m
import ROOT as r
import math
import numpy
from DetectorGeometry import DetectorGeometry
from Module import Module
import os
import Visualization
from Centroid import Centroid
from Connection import Connection

script_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
geom = DetectorGeometry("{}/data/eventdisplay_module_polygon_CMSSW_12_2_0_pre2_geom.txt".format(script_path))
cent = Centroid("{}/data/centroid_CMSSW_12_2_0_pre2.txt".format(script_path))
conn = Connection("{}/data/module_connection_tracing_CMSSW_12_2_0_pre2_merged.txt".format(script_path))

all_detids = list(geom.getDetIds(lambda x: Module(x[0]).isLower()==1))
barrelflat_detids = list(geom.getDetIds(lambda x: Module(x[0]).isBarrelFlat()==1))
reference_detid = list(geom.getDetIds(lambda x:
                                      Module(x[0]).isLower()==1 and
                                      Module(x[0]).layer()==1 and
                                      Module(x[0]).module()==1 and
                                      Module(x[0]).subdet()==5 and
                                      Module(x[0]).side()==3
                                     ))
# detid = reference_detid[0]
# reference_mod = Visualization.get_modules_goScatter3D([detid], geom)
# reference_mod['line']['color'] = 'rgb(255,0,255,0.5)'
# connected_mods = Visualization.get_modules_goScatter3D(conn.getConnection(detid), geom)
# connected_mods['line']['color'] = 'rgb(0,255,255,0.5)'
all_mods_3d = Visualization.get_modules_goScatter3D(all_detids, geom)
all_mods_xy = Visualization.get_modules_goScatterXY(all_detids, geom)
all_mods_barrelflat_xy = Visualization.get_modules_goScatterXY(barrelflat_detids, geom)
all_mods_rz = Visualization.get_modules_goScatterRZ(all_detids, geom)
# all_mods['line']['color'] = 'rgb(150,150,0,0.1)'
all_mods_3d['line']['width'] = 0.5
all_mods_xy['line']['width'] = 0.1
all_mods_rz['line']['width'] = 0.5
all_mods_barrelflat_xy['line']['width'] = 0.1
all_mods_barrelflat_xy['opacity'] = 0.1
# Visualization.create_figure([reference_mod, connected_mods], "connection.html")


f = r.TFile("LSTNtuple.root")
t = f.Get("tree")

ibadone = int(sys.argv[1])

badones = [(0, 14),
(0, 54),
(0, 100),
(0, 210),
(0, 214),
(0, 411),
(1, 136),
(1, 147),
(2, 2),
(2, 87),
(2, 123),
(2, 130),
(2, 172),
(3, 80),
(3, 81),
(4, 1),
(4, 30),
(4, 34),
(4, 80),
(4, 86),
(4, 87),
(4, 137),
(4, 153),
(4, 156),
(4, 221),
(4, 225),
(4, 249),
(4, 272),
(4, 286),
(4, 308),
(4, 314),
(4, 1064),
(5, 142),
(5, 159),
(5, 219),
(5, 227),
(5, 255),
(6, 30),
(6, 32),
(6, 33),
(6, 37),
(6, 80),
(6, 125),
(6, 150),
(6, 484),
(7, 104),
(7, 142),
(8, 12),
(8, 40),
(8, 93),
(9, 10),
(9, 20),
(9, 80),
(9, 93),
(9, 162),
(9, 173),
(9, 179),
(10, 10),
(10, 40),
(10, 45),
(10, 62),
(10, 84),
(10, 119),
(10, 186),
(10, 207),
(10, 208),
(11, 10),
(11, 26),
(11, 35),
(11, 47),
(11, 91),
(11, 112),
(12, 4),
(12, 55),
(12, 82),
(12, 151),
(13, 17),
(13, 39),
(13, 50),
(13, 53),
(13, 58),
(13, 95),
(13, 633),
(14, 2),
(14, 27),
(14, 28),
(14, 50),
(14, 51),
(14, 116),
(15, 38),
(15, 40),
(15, 165),
(16, 12),
(16, 20),
(16, 100),
(16, 105),
(16, 208),
(16, 232),
(16, 237),
(17, 0),
(17, 6),
(17, 84),
(17, 101),
(17, 130),
(17, 153),
(18, 5),
(18, 20),
(18, 23),
(18, 33),
(18, 35),
(18, 39),
(18, 63),
(18, 98),
(18, 100),
(18, 102),
(18, 107),
(18, 388),
(18, 390),
(19, 13),
(19, 82),
(19, 87),
(20, 6),
(20, 70),
(20, 87),
(20, 95),
(20, 173),
(20, 180),
(21, 71),
(21, 82),
(21, 83),
(21, 158),
(21, 198),
(21, 274),
(21, 281),
(22, 2),
(22, 3),
(22, 32),
(22, 75),
(22, 117),
(22, 132),
(22, 143),
(22, 146),
(22, 153),
(23, 10),
(23, 75),
(23, 92),
(23, 133),
(24, 34),
(24, 60),
(24, 61),
(24, 64),
(24, 77),
(24, 78),
(24, 96),
(24, 98),
(24, 100),
(24, 101),
(24, 103),
(24, 161),
(24, 197),
(24, 202),
(24, 246),
(24, 291),
(24, 292),
(24, 298),
(24, 300),
(24, 304),
(24, 1050),
(25, 11),
(25, 37),
(25, 40),
(25, 45),
(25, 68),
(25, 87),
(25, 196),
(25, 201),
(26, 28),
(26, 38),
(26, 39),
(26, 89),
(26, 145),
(26, 213),
(26, 225),
(26, 229),
(26, 257),
(26, 321),
(27, 2),
(27, 3),
(27, 88),
(27, 94),
(28, 1),
(28, 35),
(28, 113),
(28, 277),
(28, 278),
(29, 22),
(29, 33),
(29, 34),
(29, 39),
(29, 53),
(29, 57),
(29, 58),
(29, 68),
(29, 235),
(29, 305),
]

# ievt_to_display = badones[ibadone][0]
# isim_to_display = badones[ibadone][1]

ievt_to_display = 0
isim_to_display = 1

Xsim = []
Ysim = []
Zsim = []
Rsim = []
nsim = 0
recohits_xs = []
recohits_ys = []
recohits_zs = []
recohits_rs = []
simhits_xs = []
simhits_ys = []
simhits_zs = []
simhits_rs = []
detids = []

mds_xs = []
mds_ys = []
mds_zs = []
mds_rs = []

lss_lower_xs = []
lss_lower_ys = []
lss_lower_zs = []
lss_lower_rs = []
lss_upper_xs = []
lss_upper_ys = []
lss_upper_zs = []
lss_upper_rs = []

lss_xs = []
lss_ys = []
lss_zs = []
lss_rs = []

t3s_lower_xs = []
t3s_lower_ys = []
t3s_lower_zs = []
t3s_lower_rs = []
t3s_middle_xs = []
t3s_middle_ys = []
t3s_middle_zs = []
t3s_middle_rs = []
t3s_upper_xs = []
t3s_upper_ys = []
t3s_upper_zs = []
t3s_upper_rs = []

t3s_xs = []
t3s_ys = []
t3s_zs = []
t3s_rs = []

t5s_0_xs = []
t5s_0_ys = []
t5s_0_zs = []
t5s_0_rs = []
t5s_1_xs = []
t5s_1_ys = []
t5s_1_zs = []
t5s_1_rs = []
t5s_2_xs = []
t5s_2_ys = []
t5s_2_zs = []
t5s_2_rs = []
t5s_3_xs = []
t5s_3_ys = []
t5s_3_zs = []
t5s_3_rs = []
t5s_4_xs = []
t5s_4_ys = []
t5s_4_zs = []
t5s_4_rs = []

t5s_xs = []
t5s_ys = []
t5s_zs = []
t5s_rs = []

plss_0_xs = []
plss_0_ys = []
plss_0_zs = []
plss_0_rs = []
plss_1_xs = []
plss_1_ys = []
plss_1_zs = []
plss_1_rs = []
plss_2_xs = []
plss_2_ys = []
plss_2_zs = []
plss_2_rs = []
plss_3_xs = []
plss_3_ys = []
plss_3_zs = []
plss_3_rs = []

plss_xs = []
plss_ys = []
plss_zs = []
plss_rs = []

tc_xs = []
tc_ys = []
tc_zs = []
tc_rs = []

for ievt, event in enumerate(t):

    if ievt != ievt_to_display:
        continue

    for isim, _ in enumerate(event.sim_pt):

        if isim != isim_to_display:
            continue

        pt = event.sim_pt[isim]
        eta = event.sim_eta[isim]
        phi = event.sim_phi[isim]
        vx = event.sim_vx[isim]
        vy = event.sim_vy[isim]
        vz = event.sim_vz[isim]
        q = event.sim_q[isim]
        vperp = math.sqrt(vx**2 + vy**2)
        pdgid = event.sim_pdgId[isim]

        nsim += 1
        points = m.get_helix_points(m.construct_helix_from_kinematics(pt, eta, phi, vx, vy, vz, q))
        Xsim += list(points[0]) + [None] # Non is to disconnec the lines
        Ysim += list(points[1]) + [None] # Non is to disconnec the lines
        Zsim += list(points[2]) + [None] # Non is to disconnec the lines
        Rsim += list(numpy.sqrt(numpy.array(points[0])**2 + numpy.array(points[1])**2)) + [None] # Non is to disconnec the lines

        # TC
        for itc, _ in enumerate(event.sim_tcIdxAll[isim]):
            if event.sim_tcIdxAllFrac[isim][itc] > 0.75:
                tcidx = event.sim_tcIdxAll[isim][itc]
                tc_pt = event.tc_pt[tcidx]
                tc_eta = event.tc_eta[tcidx]
                tc_phi = event.tc_phi[tcidx]
                tcpoints = m.get_helix_points(m.construct_helix_from_kinematics(tc_pt, tc_eta, tc_phi, 0, 0, 0, q))
                tc_xs += list(tcpoints[0]) + [None] # Non is to disconnec the lines
                tc_ys += list(tcpoints[1]) + [None] # Non is to disconnec the lines
                tc_zs += list(tcpoints[2]) + [None] # Non is to disconnec the lines
                tc_rs += list(numpy.sqrt(numpy.array(tcpoints[0])**2 + numpy.array(tcpoints[1])**2)) + [None] # Non is to disconnec the lines

        for irecohit, _ in enumerate(event.sim_recoHitX[isim]):
            recohits_xs.append(event.sim_recoHitX[isim][irecohit])
            recohits_ys.append(event.sim_recoHitY[isim][irecohit])
            recohits_zs.append(event.sim_recoHitZ[isim][irecohit])
            recohits_rs.append(math.sqrt(event.sim_recoHitX[isim][irecohit]**2 + event.sim_recoHitY[isim][irecohit]**2))
            detid = event.sim_recoHitDetId[isim][irecohit]
            module = Module(detid)
            if module.subdet() == 4 or module.subdet() == 5:
                detids.append(event.sim_recoHitDetId[isim][irecohit])

        for isimhit, _ in enumerate(event.sim_simHitX[isim]):
            simhits_xs.append(event.sim_simHitX[isim][isimhit])
            simhits_ys.append(event.sim_simHitY[isim][isimhit])
            simhits_zs.append(event.sim_simHitZ[isim][isimhit])
            simhits_rs.append(math.sqrt(event.sim_simHitX[isim][isimhit]**2 + event.sim_simHitY[isim][isimhit]**2))
            detid = event.sim_simHitDetId[isim][isimhit]
            module = Module(detid)
            if module.subdet() == 4 or module.subdet() == 5:
                detids.append(event.sim_recoHitDetId[isim][irecohit])

        # Aggregate detids where centroid is 10cm within any of the hits in helix points and is not already part of detids list
        vicinity_detids = []
        for detid in geom.geom_data.keys():
            centroid = cent.getCentroid(detid)
            dists = numpy.sqrt((points[0] - centroid[0])**2 + (points[1] - centroid[1])**2 + (points[2] - centroid[2])**2)
            if numpy.any(dists < 5):
                if detid not in detids:
                    vicinity_detids.append(detid)

        # Aggregate detids where centroid is 10cm within any of the hits in helix points and is not already part of detids list
        looser_vicinity_detids = []
        for detid in geom.geom_data.keys():
            centroid = cent.getCentroid(detid)
            dists = numpy.sqrt((points[0] - centroid[0])**2 + (points[1] - centroid[1])**2 + (points[2] - centroid[2])**2)
            if numpy.any(dists < 30):
                if detid not in detids:
                    looser_vicinity_detids.append(detid)

        for imd, _ in enumerate(event.sim_mdIdxAll[isim]):
            if event.sim_mdIdxAllFrac[isim][imd] > 0.75:
                mdidx = event.sim_mdIdxAll[isim][imd]
                x = event.md_anchor_x[mdidx]
                y = event.md_anchor_y[mdidx]
                z = event.md_anchor_z[mdidx]
                r = math.sqrt(x**2 + y**2)
                mds_xs.append(x)
                mds_ys.append(y)
                mds_zs.append(z)
                mds_rs.append(r)

        for ils, _ in enumerate(event.sim_lsIdxAll[isim]):
            if event.sim_lsIdxAllFrac[isim][ils] > 0.75:
                lsidx = event.sim_lsIdxAll[isim][ils]
                mdidx_lower = event.ls_mdIdx0[lsidx]
                mdidx_upper = event.ls_mdIdx1[lsidx]
                x_l = event.md_anchor_x[mdidx_lower]
                y_l = event.md_anchor_y[mdidx_lower]
                z_l = event.md_anchor_z[mdidx_lower]
                r_l = math.sqrt(x_l**2 + y_l**2)
                x_u = event.md_anchor_x[mdidx_upper]
                y_u = event.md_anchor_y[mdidx_upper]
                z_u = event.md_anchor_z[mdidx_upper]
                r_u = math.sqrt(x_u**2 + y_u**2)
                lss_lower_xs.append(x_l)
                lss_lower_ys.append(y_l)
                lss_lower_zs.append(z_l)
                lss_lower_rs.append(r_l)
                lss_upper_xs.append(x_u)
                lss_upper_ys.append(y_u)
                lss_upper_zs.append(z_u)
                lss_upper_rs.append(r_u)
                lss_xs += [x_l, x_u, None]
                lss_ys += [y_l, y_u, None]
                lss_zs += [z_l, z_u, None]
                lss_rs += [r_l, r_u, None]

        for it3, _ in enumerate(event.sim_t3IdxAll[isim]):
            if event.sim_t3IdxAllFrac[isim][it3] > 0.75:
                t3idx = event.sim_t3IdxAll[isim][it3]
                lsidx_lower = event.t3_lsIdx0[t3idx]
                lsidx_upper = event.t3_lsIdx1[t3idx]
                mdidx_lower = event.ls_mdIdx0[lsidx_lower]
                mdidx_middle= event.ls_mdIdx1[lsidx_lower]
                mdidx_upper = event.ls_mdIdx1[lsidx_upper]
                x_l = event.md_anchor_x[mdidx_lower]
                y_l = event.md_anchor_y[mdidx_lower]
                z_l = event.md_anchor_z[mdidx_lower]
                r_l = math.sqrt(x_l**2 + y_l**2)
                x_m = event.md_anchor_x[mdidx_middle]
                y_m = event.md_anchor_y[mdidx_middle]
                z_m = event.md_anchor_z[mdidx_middle]
                r_m = math.sqrt(x_m**2 + y_m**2)
                x_u = event.md_anchor_x[mdidx_upper]
                y_u = event.md_anchor_y[mdidx_upper]
                z_u = event.md_anchor_z[mdidx_upper]
                r_u = math.sqrt(x_u**2 + y_u**2)
                t3s_lower_xs.append(x_l)
                t3s_lower_ys.append(y_l)
                t3s_lower_zs.append(z_l)
                t3s_lower_rs.append(r_l)
                t3s_middle_xs.append(x_m)
                t3s_middle_ys.append(y_m)
                t3s_middle_zs.append(z_m)
                t3s_middle_rs.append(r_m)
                t3s_upper_xs.append(x_u)
                t3s_upper_ys.append(y_u)
                t3s_upper_zs.append(z_u)
                t3s_upper_rs.append(r_u)
                t3s_xs += [x_l, x_m, x_u, None]
                t3s_ys += [y_l, y_m, y_u, None]
                t3s_zs += [z_l, z_m, z_u, None]
                t3s_rs += [r_l, r_m, r_u, None]

        for it5, _ in enumerate(event.sim_t5IdxAll[isim]):
            if event.sim_t5IdxAllFrac[isim][it5] > 0.75:
                t5idx = event.sim_t5IdxAll[isim][it5]
                t3idx_lower = event.t5_t3Idx0[t5idx]
                t3idx_upper = event.t5_t3Idx1[t5idx]
                lsidx_0 = event.t3_lsIdx0[t3idx_lower]
                lsidx_1 = event.t3_lsIdx1[t3idx_lower]
                lsidx_2 = event.t3_lsIdx0[t3idx_upper]
                lsidx_3 = event.t3_lsIdx1[t3idx_upper]
                mdidx_0 = event.ls_mdIdx0[lsidx_0]
                mdidx_1 = event.ls_mdIdx0[lsidx_1]
                mdidx_2 = event.ls_mdIdx0[lsidx_2]
                mdidx_3 = event.ls_mdIdx0[lsidx_3]
                mdidx_4 = event.ls_mdIdx1[lsidx_3]
                x_0 = event.md_anchor_x[mdidx_0]
                y_0 = event.md_anchor_y[mdidx_0]
                z_0 = event.md_anchor_z[mdidx_0]
                r_0 = math.sqrt(x_0**2 + y_0**2)
                x_1 = event.md_anchor_x[mdidx_1]
                y_1 = event.md_anchor_y[mdidx_1]
                z_1 = event.md_anchor_z[mdidx_1]
                r_1 = math.sqrt(x_1**2 + y_1**2)
                x_2 = event.md_anchor_x[mdidx_2]
                y_2 = event.md_anchor_y[mdidx_2]
                z_2 = event.md_anchor_z[mdidx_2]
                r_2 = math.sqrt(x_2**2 + y_2**2)
                x_3 = event.md_anchor_x[mdidx_3]
                y_3 = event.md_anchor_y[mdidx_3]
                z_3 = event.md_anchor_z[mdidx_3]
                r_3 = math.sqrt(x_3**2 + y_3**2)
                x_4 = event.md_anchor_x[mdidx_4]
                y_4 = event.md_anchor_y[mdidx_4]
                z_4 = event.md_anchor_z[mdidx_4]
                r_4 = math.sqrt(x_4**2 + y_4**2)
                t5s_0_xs.append(x_0)
                t5s_0_ys.append(y_0)
                t5s_0_zs.append(z_0)
                t5s_0_rs.append(r_0)
                t5s_1_xs.append(x_1)
                t5s_1_ys.append(y_1)
                t5s_1_zs.append(z_1)
                t5s_1_rs.append(r_1)
                t5s_2_xs.append(x_2)
                t5s_2_ys.append(y_2)
                t5s_2_zs.append(z_2)
                t5s_2_rs.append(r_2)
                t5s_3_xs.append(x_3)
                t5s_3_ys.append(y_3)
                t5s_3_zs.append(z_3)
                t5s_3_rs.append(r_3)
                t5s_4_xs.append(x_4)
                t5s_4_ys.append(y_4)
                t5s_4_zs.append(z_4)
                t5s_4_rs.append(r_4)
                t5s_xs += [x_0, x_1, x_2, x_3, x_4, None]
                t5s_ys += [y_0, y_1, y_2, y_3, y_4, None]
                t5s_zs += [z_0, z_1, z_2, z_3, z_4, None]
                t5s_rs += [r_0, r_1, r_2, r_3, r_4, None]

        for ipls, _ in enumerate(event.sim_plsIdxAll[isim]):
            if event.sim_plsIdxAllFrac[isim][ipls] > 0.75:
                plsidx = event.sim_plsIdxAll[isim][ipls]
                x_0 = event.pls_hit0_x[plsidx]
                y_0 = event.pls_hit0_y[plsidx]
                z_0 = event.pls_hit0_z[plsidx]
                r_0 = math.sqrt(x_0**2 + y_0**2)
                x_1 = event.pls_hit1_x[plsidx]
                y_1 = event.pls_hit1_y[plsidx]
                z_1 = event.pls_hit1_z[plsidx]
                r_1 = math.sqrt(x_1**2 + y_1**2)
                x_2 = event.pls_hit2_x[plsidx]
                y_2 = event.pls_hit2_y[plsidx]
                z_2 = event.pls_hit2_z[plsidx]
                r_2 = math.sqrt(x_2**2 + y_2**2)
                x_3 = event.pls_hit3_x[plsidx]
                y_3 = event.pls_hit3_y[plsidx]
                z_3 = event.pls_hit3_z[plsidx]
                r_3 = math.sqrt(x_3**2 + y_3**2)
                plss_0_xs.append(x_0)
                plss_0_ys.append(y_0)
                plss_0_zs.append(z_0)
                plss_0_rs.append(r_0)
                plss_1_xs.append(x_1)
                plss_1_ys.append(y_1)
                plss_1_zs.append(z_1)
                plss_1_rs.append(r_1)
                plss_2_xs.append(x_2)
                plss_2_ys.append(y_2)
                plss_2_zs.append(z_2)
                plss_2_rs.append(r_2)
                plss_3_xs.append(x_3)
                plss_3_ys.append(y_3)
                plss_3_zs.append(z_3)
                plss_3_rs.append(r_3)
                if x_3 != -999:
                    plss_xs += [x_0, x_1, x_2, x_3, None]
                    plss_ys += [y_0, y_1, y_2, y_3, None]
                    plss_zs += [z_0, z_1, z_2, z_3, None]
                    plss_rs += [r_0, r_1, r_2, r_3, None]
                else:
                    plss_xs += [x_0, x_1, x_2, None]
                    plss_ys += [y_0, y_1, y_2, None]
                    plss_zs += [z_0, z_1, z_2, None]
                    plss_rs += [r_0, r_1, r_2, None]

print("All parsed")


modules_to_draw = Visualization.get_modules_goScatter3D(detids, geom)
modules_to_draw['line']['color'] = 'orange'
modules_to_draw['line']['width'] = 0.5
modules_to_draw_xy = Visualization.get_modules_goScatterXY(detids, geom)
modules_to_draw_xy['line']['color'] = 'orange'
modules_to_draw_xy['line']['width'] = 0.5
modules_to_draw_rz = Visualization.get_modules_goScatterRZ(detids, geom)
modules_to_draw_rz['line']['color'] = 'orange'
modules_to_draw_rz['line']['width'] = 0.5

modules_filled_to_draw = Visualization.get_modules_filled_goScatter3D(detids, geom)
vicinity_modules_filled_to_draw = Visualization.get_modules_filled_goScatter3D(vicinity_detids, geom)
for vmod in vicinity_modules_filled_to_draw:
    vmod['surfacecolor'] = "cyan"
    vmod['line']['color'] = "cyan"
# all_modules_filled_to_draw = Visualization.get_modules_filled_goScatter3D(all_detids, geom)
looser_vicinity_modules_to_draw = Visualization.get_modules_goScatter3D(looser_vicinity_detids, geom)
looser_vicinity_modules_to_draw['line']['width'] = 0.1

sims_to_draw = go.Scatter3d(
        x = Zsim,
        y = Ysim,
        z = Xsim,
        mode='lines',
        line=dict(
            color='red',
            width=2,
        ),
        opacity=1,
        hoverinfo='none',
        name='Sim Track',
)

tcs_to_draw = go.Scatter3d(
        x = tc_zs,
        y = tc_ys,
        z = tc_xs,
        mode='lines',
        line=dict(
            color='yellow',
            width=2,
        ),
        opacity=1,
        hoverinfo='none',
        name='Track Candidate',
)

recohits_to_draw = go.Scatter3d(x=recohits_zs,
                            y=recohits_ys,
                            z=recohits_xs,
                            mode="markers",
                            marker=dict(
                                symbol='circle',
                                size=2.5,
                                color='cyan',
                                colorscale='Viridis',
                                opacity=0.8,
                                ),
                            hoverinfo='text',
                            name='Reco Hit',
                           )

simhits_to_draw = go.Scatter3d(x=simhits_zs,
                            y=simhits_ys,
                            z=simhits_xs,
                            mode="markers",
                            marker=dict(
                                symbol='circle',
                                size=2.5,
                                color='magenta',
                                colorscale='Viridis',
                                opacity=0.8,
                                ),
                            hoverinfo='text',
                            name='Sim Hit',
                           )

mds_to_draw = go.Scatter3d(x=mds_zs,
                            y=mds_ys,
                            z=mds_xs,
                            mode="markers",
                            marker=dict(
                                symbol='circle',
                                size=3.5,
                                color='yellow',
                                colorscale='Viridis',
                                opacity=0.8,
                                ),
                            hoverinfo='text',
                            name='Mini-Doublets',
                           )

lss_to_draw = go.Scatter3d(
        x = lss_zs,
        y = lss_ys,
        z = lss_xs,
        mode='lines',
        line=dict(
            color='green',
            width=4,
        ),
        opacity=1,
        hoverinfo='none',
        name='Line Segments',
)

t3s_to_draw = go.Scatter3d(
        x = t3s_zs,
        y = t3s_ys,
        z = t3s_xs,
        mode='lines',
        line=dict(
            color='blue',
            width=4,
        ),
        opacity=1,
        hoverinfo='none',
        name='Triplets',
)

t5s_to_draw = go.Scatter3d(
        x = t5s_zs,
        y = t5s_ys,
        z = t5s_xs,
        mode='lines',
        line=dict(
            color='purple',
            width=4,
        ),
        opacity=1,
        hoverinfo='none',
        name='Quintuplets',
)

plss_to_draw = go.Scatter3d(
        x = plss_zs,
        y = plss_ys,
        z = plss_xs,
        mode='lines',
        line=dict(
            color='orange',
            width=4,
        ),
        opacity=1,
        hoverinfo='none',
        name='Pixel LS',
)

plss_hits_to_draw = go.Scatter3d(
        x = plss_zs,
        y = plss_ys,
        z = plss_xs,
        mode="markers",
        marker=dict(
            symbol='circle',
            size=2.5,
            color='cyan',
            colorscale='Viridis',
            opacity=0.8,
            ),
        hoverinfo='text',
        name='Inner Tracker Hits',
)

fig = go.Figure([sims_to_draw, tcs_to_draw, plss_hits_to_draw, recohits_to_draw, simhits_to_draw, mds_to_draw, lss_to_draw, t3s_to_draw, t5s_to_draw, plss_to_draw, looser_vicinity_modules_to_draw] + modules_filled_to_draw + vicinity_modules_filled_to_draw)

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgb(0,0,0,1)",
    scene = dict(
        xaxis = dict(nticks=10, range=[-300,300],),
        yaxis = dict(nticks=10, range=[-200,200],),
        zaxis = dict(nticks=10, range=[-200,200],),
        aspectratio=dict(x=1, y=0.666, z=0.666),
    ),
    width=1400,
    height=800,
    margin=dict(r=20, l=10, b=10, t=10));

fig.write_html("htmls/sim_{}_all.html".format(ibadone))

sim_to_draw_xy = go.Scatter(
        x = Xsim,
        y = Ysim,
        mode='lines',
        line=dict(
            color='red',
            width=2.0,
        ),
        opacity=1,
        hoverinfo='none',
        name='Sim Track',
)

tcs_to_draw_xy = go.Scatter(
        x = tc_xs,
        y = tc_ys,
        mode='lines',
        line=dict(
            color='yellow',
            width=2.0,
        ),
        opacity=1,
        hoverinfo='none',
        name='Track Candidates',
)

recohits_to_draw_xy = go.Scatter(x=recohits_xs,
                             y=recohits_ys,
                             mode="markers",
                             marker=dict(
                                 symbol='circle',
                                 size=4.5,
                                 color='cyan',
                                 colorscale='Viridis',
                                 opacity=0.8,
                                 ),
                             hoverinfo='text',
                             name='Reco Hit',
                            )

simhits_to_draw_xy = go.Scatter(x=simhits_xs,
                             y=simhits_ys,
                             mode="markers",
                             marker=dict(
                                 symbol='circle',
                                 size=4.5,
                                 color='magenta',
                                 colorscale='Viridis',
                                 opacity=0.8,
                                 ),
                             hoverinfo='text',
                             name='Sim Hit',
                            )

mds_to_draw_xy = go.Scatter(x=mds_xs,
                             y=mds_ys,
                             mode="markers",
                             marker=dict(
                                 symbol='circle',
                                 size=6.5,
                                 color='yellow',
                                 colorscale='Viridis',
                                 opacity=0.8,
                                 ),
                             hoverinfo='text',
                             name='Mini-Doublets',
                            )

lss_to_draw_xy = go.Scatter(
        x = lss_xs,
        y = lss_ys,
        mode='lines',
        line=dict(
            color='green',
            width=3,
        ),
        opacity=1,
        hoverinfo='none',
        name='Line Segments',
)

t3s_to_draw_xy = go.Scatter(
        x = t3s_xs,
        y = t3s_ys,
        mode='lines',
        line=dict(
            color='blue',
            width=3,
        ),
        opacity=1,
        hoverinfo='none',
        name='Triplets',
)

t5s_to_draw_xy = go.Scatter(
        x = t5s_xs,
        y = t5s_ys,
        mode='lines',
        line=dict(
            color='purple',
            width=3,
        ),
        opacity=1,
        hoverinfo='none',
        name='Quintuplets',
)

plss_to_draw_xy = go.Scatter(
        x = plss_xs,
        y = plss_ys,
        mode='lines',
        line=dict(
            color='orange',
            width=3,
        ),
        opacity=1,
        hoverinfo='none',
        name='Pixel LS',
)

plss_hits_to_draw = go.Scatter(
        x = plss_xs,
        y = plss_ys,
        mode="markers",
        marker=dict(
            symbol='circle',
            size=4.5,
            color='cyan',
            colorscale='Viridis',
            opacity=0.8,
            ),
        hoverinfo='text',
        name='Inner Tracker Hits',
)

vicinity_modules_xy = Visualization.get_modules_goScatterXY(vicinity_detids, geom)
vicinity_modules_xy['line']['width'] = 0.5

fig = go.Figure([sim_to_draw_xy, tcs_to_draw_xy, recohits_to_draw_xy, plss_hits_to_draw, simhits_to_draw_xy, mds_to_draw_xy, all_mods_barrelflat_xy, vicinity_modules_xy, modules_to_draw_xy, lss_to_draw_xy, t3s_to_draw_xy, t5s_to_draw_xy, plss_to_draw_xy])
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgb(0,0,0,1)",
    xaxis_range=[-130,130],
    yaxis_range=[-150,150],
    width=800,
    height=800,
    margin=dict(r=20, l=10, b=10, t=10));
fig.write_html("htmls/sim_{}_all_xy.html".format(ibadone))


sim_to_draw_rz = go.Scatter(
        x = Zsim,
        y = Rsim,
        mode='lines',
        line=dict(
            color='red',
            width=2.0,
        ),
        opacity=1,
        hoverinfo='none',
        name='Sim Track',
)

tcs_to_draw_rz = go.Scatter(
        x = tc_zs,
        y = tc_rs,
        mode='lines',
        line=dict(
            color='yellow',
            width=2.0,
        ),
        opacity=1,
        hoverinfo='none',
        name='Track Candidates',
)

recohits_to_draw_rz = go.Scatter(x=recohits_zs,
                             y=recohits_rs,
                             mode="markers",
                             marker=dict(
                                 symbol='circle',
                                 size=4.5,
                                 color='cyan',
                                 colorscale='Viridis',
                                 opacity=0.8,
                                 ),
                             hoverinfo='text',
                             name='Reco Hit',
                            )

simhits_to_draw_rz = go.Scatter(x=simhits_zs,
                             y=simhits_rs,
                             mode="markers",
                             marker=dict(
                                 symbol='circle',
                                 size=4.5,
                                 color='magenta',
                                 colorscale='Viridis',
                                 opacity=0.8,
                                 ),
                             hoverinfo='text',
                             name='Sim Hit',
                            )

mds_to_draw_rz = go.Scatter(x=mds_zs,
                             y=mds_rs,
                             mode="markers",
                             marker=dict(
                                 symbol='circle',
                                 size=12.5,
                                 color='yellow',
                                 colorscale='Viridis',
                                 opacity=0.8,
                                 ),
                             hoverinfo='text',
                             name='Mini-Doublets',
                            )

lss_to_draw_rz = go.Scatter(
        x = lss_zs,
        y = lss_rs,
        mode='lines',
        line=dict(
            color='green',
            width=3,
        ),
        opacity=1,
        hoverinfo='none',
        name='Line Segments',
)

t3s_to_draw_rz = go.Scatter(
        x = t3s_zs,
        y = t3s_rs,
        mode='lines',
        line=dict(
            color='blue',
            width=3,
        ),
        opacity=1,
        hoverinfo='none',
        name='Triplets',
)

t5s_to_draw_rz = go.Scatter(
        x = t5s_zs,
        y = t5s_rs,
        mode='lines',
        line=dict(
            color='purple',
            width=3,
        ),
        opacity=1,
        hoverinfo='none',
        name='Quintuplets',
)

plss_to_draw_rz = go.Scatter(
        x = plss_zs,
        y = plss_rs,
        mode='lines',
        line=dict(
            color='orange',
            width=3,
        ),
        opacity=1,
        hoverinfo='none',
        name='Pixel LS',
)

plss_hits_to_draw_rz = go.Scatter(x=plss_zs,
                             y=plss_rs,
                             mode="markers",
                             marker=dict(
                                 symbol='circle',
                                 size=4.5,
                                 color='cyan',
                                 colorscale='Viridis',
                                 opacity=0.8,
                                 ),
                             hoverinfo='text',
                             name='Inner Tracker Hits',
                            )

# eta lines
etas = numpy.linspace(0, 3.2, num=int(3.2/0.2)+1)
thetas = 2*numpy.arctan(numpy.exp(-etas))
slopes = numpy.sin(thetas) / numpy.cos(thetas)
rs_etalines = []
zs_etalines = []
for slope in slopes:
    r = 120
    z = r / slope
    rs_etalines += [0, r, None]
    zs_etalines += [0, z, None]
    rs_etalines += [0, r, None]
    zs_etalines += [0, -z, None]

eta_guidelines_rz = go.Scatter(
        x = zs_etalines,
        y = rs_etalines,
        mode='lines',
        line=dict(
            color='gray',
            width=1,
        ),
        opacity=0.4,
        hoverinfo='none',
        name='Eta guidelines',
)

track_information = "pt={:.2f}, eta={:.2f}, phi={:.2f}, vx={:.2f}, vy={:.2f}, vz={:.2f}, vperp={:.2f}, pdgid={}, charge={}".format(pt, eta, phi, vx, vy, vz, vperp, pdgid, q)
dis_x = 0.2
dis_y = 0.02

fig = go.Figure([eta_guidelines_rz, modules_to_draw_rz, all_mods_rz, sim_to_draw_rz, tcs_to_draw_rz, lss_to_draw_rz, t3s_to_draw_rz, t5s_to_draw_rz, plss_to_draw_rz, simhits_to_draw_rz, recohits_to_draw_rz, mds_to_draw_rz, plss_hits_to_draw])
fig.add_annotation(text=track_information, xref="paper", yref="paper", x=dis_x, y=dis_y, showarrow=False)
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgb(0,0,0,1)",
    xaxis_range=[-300,300],
    yaxis_range=[0,120],
    width=1200,
    height=600,
    margin=dict(r=20, l=10, b=10, t=10));
fig.write_html("htmls/sim_{}_all_rz.html".format(ibadone))

fig = go.Figure([eta_guidelines_rz, modules_to_draw_rz, all_mods_rz, sim_to_draw_rz, tcs_to_draw_rz, lss_to_draw_rz, t3s_to_draw_rz, t5s_to_draw_rz, plss_to_draw_rz, simhits_to_draw_rz, recohits_to_draw_rz, mds_to_draw_rz, plss_hits_to_draw])
fig.add_annotation(text=track_information, xref="paper", yref="paper", x=dis_x, y=dis_y, showarrow=False)
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgb(0,0,0,1)",
    xaxis_range=[0,300],
    yaxis_range=[0,120],
    width=1200,
    height=600,
    margin=dict(r=20, l=10, b=10, t=10));
fig.write_html("htmls/sim_{}_all_pos_rz.html".format(ibadone))


fig = go.Figure([eta_guidelines_rz, modules_to_draw_rz, all_mods_rz, sim_to_draw_rz, tcs_to_draw_rz, lss_to_draw_rz, t3s_to_draw_rz, t5s_to_draw_rz, plss_to_draw_rz, simhits_to_draw_rz, recohits_to_draw_rz, mds_to_draw_rz, plss_hits_to_draw])
fig.add_annotation(text=track_information, xref="paper", yref="paper", x=dis_x, y=dis_y, showarrow=False)
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgb(0,0,0,1)",
    xaxis_range=[-300,0],
    yaxis_range=[0,120],
    width=1200,
    height=600,
    margin=dict(r=20, l=10, b=10, t=10));
fig.write_html("htmls/sim_{}_all_neg_rz.html".format(ibadone))


