#!/bin/env python

import plotly
import plotly.graph_objs as go
import plotly.express as px
import numpy
import math
import LSTMath as m
import os
from DetectorGeometry import DetectorGeometry
from Centroid import Centroid
from Connection import Connection
from Module import Module

# Open the geometry related files (module locations and centroid and connections)
script_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
geom = DetectorGeometry("{}/data/eventdisplay_module_polygon_CMSSW_12_2_0_pre2_geom.txt".format(script_path))
cent = Centroid("{}/data/centroid_CMSSW_12_2_0_pre2.txt".format(script_path))
conn = Connection("{}/data/module_connection_tracing_CMSSW_12_2_0_pre2_merged.txt".format(script_path))

def create_figure(objs, name):
    fig = go.Figure(objs)
    size = 800
    fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgb(0,0,0,1)",
            scene = dict(
                xaxis = dict(nticks=10, range=[-300,300],),
                yaxis = dict(nticks=10, range=[-300,300],),
                zaxis = dict(nticks=10, range=[-300,300],),
                aspectratio=dict(x=1, y=1, z=1),
                ),
            width=1000,
            height=1000,
            margin=dict(r=20, l=10, b=10, t=10));
    fig.write_html(name)

def create_3d_figures(objs_to_draw, output_name):
    # Draw the figure
    fig = go.Figure(objs_to_draw)

    # Adjust the viewing setup
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

    # Write the graph
    fig.write_html(output_name)

def get_modules_goScatter3D(list_of_detids, geom, showlegend=True):
    zs=[]
    xs=[]
    ys=[]
    j = geom.geom_data
    for detid in list_of_detids:
        z_=[j[detid][0][0], j[detid][1][0], j[detid][2][0], j[detid][3][0], j[detid][0][0], None]
        x_=[j[detid][0][1], j[detid][1][1], j[detid][2][1], j[detid][3][1], j[detid][0][1], None]
        y_=[j[detid][0][2], j[detid][1][2], j[detid][2][2], j[detid][3][2], j[detid][0][2], None]
        zs += z_
        xs += x_
        ys += y_
    mods = go.Scatter3d(x=zs,
                        y=ys,
                        z=xs,
                        mode="lines",
                        line=dict(
                            color="rgb(0,255,255,0.5)",
                            ),
                        hoverinfo='text',
                        name='Module',
                        showlegend=showlegend,
                        )
    return mods

def get_modules_filled_goScatter3D(list_of_detids, geom, showlegend=True):
    j = geom.geom_data
    rtn_mods = []
    for detid in list_of_detids:
        zs=[]
        xs=[]
        ys=[]
        z_=[j[detid][0][0], j[detid][1][0], j[detid][2][0], j[detid][3][0], j[detid][0][0], None]
        x_=[j[detid][0][1], j[detid][1][1], j[detid][2][1], j[detid][3][1], j[detid][0][1], None]
        y_=[j[detid][0][2], j[detid][1][2], j[detid][2][2], j[detid][3][2], j[detid][0][2], None]
        zs += z_
        xs += x_
        ys += y_
        mod = Module(detid)
        layer = mod.logicalLayer()
        isLower =  mod.isLower()
        mods = go.Scatter3d(x=zs,
                            y=ys,
                            z=xs,
                            mode="lines",
                            line=dict(
                                color="rgb(205,115,0,0.5)",
                                width=0.5,
                                ),
                            hoverinfo='text',
                            name='Module Layer={} Lower={}'.format(layer, isLower),
                            surfacecolor='rgb(205,115,0,0.1)',
                            surfaceaxis=1,
                            opacity=1,
                            showlegend=showlegend,
                            )
        rtn_mods.append(mods)
    return rtn_mods

def get_modules_goScatterXY(list_of_detids, geom):
    zs=[]
    xs=[]
    ys=[]
    j = geom.geom_data
    for detid in list_of_detids:
        z_=[j[detid][0][0], j[detid][1][0], j[detid][2][0], j[detid][3][0], j[detid][0][0], None]
        x_=[j[detid][0][1], j[detid][1][1], j[detid][2][1], j[detid][3][1], j[detid][0][1], None]
        y_=[j[detid][0][2], j[detid][1][2], j[detid][2][2], j[detid][3][2], j[detid][0][2], None]
        zs += z_
        xs += x_
        ys += y_
    mods = go.Scatter(x=xs,
                     y=ys,
                     mode="lines",
                     line=dict(
                         color="rgb(0,255,255,0.5)",
                         ),
                     hoverinfo='text',
                     name='Module',
                     # fill="toself",
                     )
    return mods

def get_modules_goScatterRZ(list_of_detids, geom):
    zs=[]
    xs=[]
    ys=[]
    rs=[]
    j = geom.geom_data
    for detid in list_of_detids:
        zz_ = [j[detid][0][0], j[detid][1][0], j[detid][2][0], j[detid][3][0], j[detid][0][0]]
        yy_ = [j[detid][0][1], j[detid][1][1], j[detid][2][1], j[detid][3][1], j[detid][0][1]]
        xx_ = [j[detid][0][2], j[detid][1][2], j[detid][2][2], j[detid][3][2], j[detid][0][2]]
        rr_ = list(numpy.sqrt(numpy.array(xx_)**2 + numpy.array(yy_)**2))
        z_= zz_ + [None]
        x_= zz_ + [None]
        y_= zz_ + [None]
        r_= rr_ + [None]
        zs += z_
        xs += x_
        ys += y_
        rs += r_
    mods = go.Scatter(x=zs,
                     y=rs,
                     mode="lines",
                     line=dict(
                         color="rgb(0,255,255,0.5)",
                         ),
                     hoverinfo='text',
                     name='Module',
                     )
    return mods

def get_lss_goScatter3D(lss_xs, lss_ys, lss_zs, lss_idxs):
    lss_to_draw = []
    for ls_xs, ls_ys, ls_zs, ls_idx in zip(lss_xs, lss_ys, lss_zs, lss_idxs):
        ls_to_draw = go.Scatter3d(
                x=ls_zs,
                y=ls_ys,
                z=ls_xs,
                mode='lines',
                line=dict(
                    color='green',
                    width=2,
                    ),
                opacity=1,
                hoverinfo='none',
                name='Line Segment {}'.format(ls_idx),
                )
        lss_to_draw.append(ls_to_draw)
    return lss_to_draw


class DisplaySimTrack:

    def __init__(self, tree, ievt_to_display, isim_to_display):
        self.ievt = ievt_to_display
        self.isim = isim_to_display
        self.get_simtrk_infos(tree, ievt_to_display, isim_to_display)

    def get_simtrk_infos(self, tree, ievt_to_display, isim_to_display):
        # get the event of interest
        for ievt, event in enumerate(tree):
            if ievt == ievt_to_display:
                break

        # track of interest
        pt = event.sim_pt[isim_to_display]
        eta = event.sim_eta[isim_to_display]
        phi = event.sim_phi[isim_to_display]
        vx = event.sim_vx[isim_to_display]
        vy = event.sim_vy[isim_to_display]
        vz = event.sim_vz[isim_to_display]
        q = event.sim_q[isim_to_display]
        vperp = math.sqrt(vx**2 + vy**2)
        pdgid = event.sim_pdgId[isim_to_display]

        # get the parametric x, y, z, r positions (these are points sampled from the helix with small steps of parametrization)
        points = m.get_helix_points(m.construct_helix_from_kinematics(pt, eta, phi, vx, vy, vz, q))
        self.sim_xs = list(points[0])
        self.sim_ys = list(points[1])
        self.sim_zs = list(points[2])
        self.sim_rs = list(numpy.sqrt(numpy.array(points[0])**2 + numpy.array(points[1])**2))

        # simhits
        self.simhits_xs = list(event.sim_simHitX[isim_to_display])
        self.simhits_ys = list(event.sim_simHitY[isim_to_display])
        self.simhits_zs = list(event.sim_simHitZ[isim_to_display])
        self.simhits_rs = list(numpy.sqrt(numpy.array(self.simhits_xs)**2 + numpy.array(self.simhits_ys)**2))

        # recohits
        self.recohits_xs = list(event.sim_recoHitX[isim_to_display])
        self.recohits_ys = list(event.sim_recoHitY[isim_to_display])
        self.recohits_zs = list(event.sim_recoHitZ[isim_to_display])
        self.recohits_rs = list(numpy.sqrt(numpy.array(self.recohits_xs)**2 + numpy.array(self.recohits_ys)**2))

        # mini-doublets
        self.mds_xs = []
        self.mds_ys = []
        self.mds_zs = []
        self.mds_rs = []
        for imd, _ in enumerate(event.sim_mdIdxAll[isim_to_display]):
            if event.sim_mdIdxAllFrac[isim_to_display][imd] > 0.75:
                mdidx = event.sim_mdIdxAll[isim_to_display][imd]
                x = event.md_anchor_x[mdidx]
                y = event.md_anchor_y[mdidx]
                z = event.md_anchor_z[mdidx]
                r = math.sqrt(x**2 + y**2)
                self.mds_xs.append(x)
                self.mds_ys.append(y)
                self.mds_zs.append(z)
                self.mds_rs.append(r)

        # line-segments
        self.lss_xs = []
        self.lss_ys = []
        self.lss_zs = []
        self.lss_rs = []
        self.lss_idxs = []
        for ils, _ in enumerate(event.sim_lsIdxAll[isim_to_display]):
            if event.sim_lsIdxAllFrac[isim_to_display][ils] > 0.75:
                lsidx = event.sim_lsIdxAll[isim_to_display][ils]
                mdidx0 = event.ls_mdIdx0[lsidx]
                mdidx1 = event.ls_mdIdx1[lsidx]
                x0 = event.md_anchor_x[mdidx0]
                y0 = event.md_anchor_y[mdidx0]
                z0 = event.md_anchor_z[mdidx0]
                r0 = math.sqrt(x0**2 + y0**2)
                x1 = event.md_anchor_x[mdidx1]
                y1 = event.md_anchor_y[mdidx1]
                z1 = event.md_anchor_z[mdidx1]
                r1 = math.sqrt(x1**2 + y1**2)
                self.lss_xs.append([x0, x1])
                self.lss_ys.append([y0, y1])
                self.lss_zs.append([z0, z1])
                self.lss_rs.append([r0, r1])
                self.lss_idxs.append(lsidx)

        # detids
        self.reco_detids = list(event.sim_recoHitDetId[isim_to_display])

        # Get list of logical layers with a reco hit
        self.logical_layers = []
        for detid in self.reco_detids:
            ll = Module(detid).logicalLayer()
            if ll not in self.logical_layers:
                self.logical_layers.append(ll)

        # Aggregate detids where centroid is 10cm within any of the hits in helix points and is not already part of detids list and also if a hit doesn't exist in that logical layer
        self.vicinity_detids = []
        for detid in geom.geom_data.keys():
            centroid = cent.getCentroid(detid)
            dists = numpy.sqrt((points[0] - centroid[0])**2 + (points[1] - centroid[1])**2 + (points[2] - centroid[2])**2)
            if numpy.any(dists < 5):
                if detid not in self.reco_detids and Module(detid).logicalLayer() not in self.logical_layers:
                    self.vicinity_detids.append(detid)

        # Aggregate detids where centroid is 10cm within any of the hits in helix points and is not already part of detids list
        self.looser_vicinity_detids = []
        for detid in geom.geom_data.keys():
            centroid = cent.getCentroid(detid)
            dists = numpy.sqrt((points[0] - centroid[0])**2 + (points[1] - centroid[1])**2 + (points[2] - centroid[2])**2)
            if numpy.any(dists < 20):
                if detid not in self.reco_detids and detid not in self.vicinity_detids:
                    self.looser_vicinity_detids.append(detid)

    def get_simtrk_goScatter3Ds(self):

        # graphical object of the simulated track
        sims_to_draw = go.Scatter3d(
                x=self.sim_zs,
                y=self.sim_ys,
                z=self.sim_xs,
                mode='lines',
                line=dict(
                    color='red',
                    width=2,
                    ),
                opacity=1,
                hoverinfo='none',
                name='Sim Track',
                )

        simhits_to_draw = go.Scatter3d(
                x=self.simhits_zs,
                y=self.simhits_ys,
                z=self.simhits_xs,
                mode="markers",
                marker=dict(
                    symbol='circle-open',
                    size=3.5,
                    color='magenta',
                    colorscale='Viridis',
                    opacity=0.8,
                    ),
                hoverinfo='text',
                name='Sim Hit',
                )

        recohits_to_draw = go.Scatter3d(
                x=self.recohits_zs,
                y=self.recohits_ys,
                z=self.recohits_xs,
                mode="markers",
                marker=dict(
                    symbol='circle',
                    size=3.5,
                    color='cyan',
                    colorscale='Viridis',
                    opacity=0.8,
                    ),
                # hoverinfo='text',
                name='Reco Hit',
                )

        mds_to_draw = go.Scatter3d(
                x=self.mds_zs,
                y=self.mds_ys,
                z=self.mds_xs,
                mode="markers",
                marker=dict(
                    symbol='diamond',
                    size=3.5,
                    color='yellow',
                    colorscale='Viridis',
                    opacity=0.8,
                    ),
                # hoverinfo='text',
                name='Mini-Doublets',
                )

        modules_to_draw = get_modules_filled_goScatter3D(self.reco_detids, geom)
        vicinity_modules_to_draw = get_modules_filled_goScatter3D(self.vicinity_detids, geom)
        for obj in vicinity_modules_to_draw:
            obj['line']['color'] = 'gray'
            obj['surfacecolor'] = 'gray'
            obj['opacity'] = 1
        looser_vicinity_modules_to_draw = get_modules_goScatter3D(self.looser_vicinity_detids, geom)
        looser_vicinity_modules_to_draw['line']['color'] = 'rgb(50,50,50,1)'
        looser_vicinity_modules_to_draw['line']['width'] = 0.4
        looser_vicinity_modules_to_draw['name'] = 'Extra Modules'

        lss_to_draw = get_lss_goScatter3D(self.lss_xs, self.lss_ys, self.lss_zs, self.lss_idxs)

        return [looser_vicinity_modules_to_draw] + modules_to_draw + vicinity_modules_to_draw + [sims_to_draw, simhits_to_draw, recohits_to_draw, mds_to_draw] + lss_to_draw

