"""interface.py: The web interface for Tessellation Drawing."""

__author__      = "Luyi HUAANG <luyi.lancle.huaang@gmail.com>"
__copyright__   = "Copyright 2020"

from flask import Flask, jsonify, abort, make_response, \
    url_for, request as flask_request, \
        after_this_request
from flask_restful import Resource, Api
import requests
import numpy as np
import pandas as pd
import json

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash import Dash
from dash.dependencies import Input, Output, State

from dash_table import DataTable
from dash.exceptions import PreventUpdate
from datetime import datetime
# import plotly.figure_factory as ff
import plotly.express as px

import plotly
# import chart_studio.plotly as py
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State

# import networkx as nx
import pprint
import re
import os
import random

from dash_canvas import DashCanvas
from PIL import Image, ImageChops
import base64
import sys
from dash.dependencies import Input, Output
# from model_interface import ModelUtility
from assist import MainFuncAssistance, to_json, to_json_str

from dash_canvas.utils import array_to_data_url, \
    parse_jsonstring, image_string_to_PILImage
from io import BytesIO
from endpoint_action import EndpointAction
from collections import namedtuple
from glob import glob

class TessellationRendering():
    def add_all_endpoints(self):
        # Add root endpoint
        self.add_endpoint(endpoint="/api", endpoint_name="/api", handler=self.action)
        self.add_endpoint(endpoint="/api/canvas", endpoint_name="/api/canvas", handler=self.intfImgReco)

    def add_endpoint(self, endpoint=None, endpoint_name=None, handler=None):
        self.server_app.add_url_rule(endpoint, endpoint_name, EndpointAction(handler))
        print(f"Added url rule: {endpoint}, {endpoint_name}, {handler}")
        # You can also add options here : "... , methods=['POST'], ... "

    def action(self):
        # Dummy action
        return "this is sample action based on" # String that will be returned and display on the webpage


    def __init__(self, task_conf, cf_logger):

        self.logger = cf_logger

        self.rendering_server = task_conf["user_interface"]\
            ["web"]["rendering_server"]

        self.conf_render_dash_apps = task_conf["user_interface"]\
            ["web"]["render_dash_apps"]

        self.conf_update_dash_apps = task_conf["user_interface"]\
            ["web"]["update_dash_apps"]

        self.conf_dapp_homepage = task_conf["user_interface"]\
            ["web"]["conf_homepage"]

        self.plotly_config = task_conf["user_interface"]\
            ["web"]["plotly_config"]

        self.conf_case = task_conf["conf_case"]
        assert "case_dir_path" in self.conf_case,\
            "Key \"case_dir_path\" is missing in case config."
        self.case_dir_path = self.conf_case["case_dir_path"]
        assert "conf_model" in self.conf_case, \
            f"Key \"conf_model\" in not in config"

        # self.cnnc_tile = ModelUtility(self.conf_case["conf_model"], \
        #     self.conf_case["case_dir_path"])
        # self.cnnc_panel = ModelUtility(self.conf_case["conf_model"], \
        #     self.conf_case["case_dir_path"], "model_frame_reco")
        self.conf_model_tile = \
            self.conf_case["conf_model"]["model_tile_reco"]
        self.conf_model_frame = \
            self.conf_case["conf_model"]["model_frame_reco"]

        self.initializeDASH()
        self.add_all_endpoints()

    def initializeDASH(self):
        self.server_app = Flask(__name__)

        self.dash_apps = {}

        self.app = Dash(
            server=self.server_app,
            url_base_pathname=self.rendering_server["base"],
            external_stylesheets=[dbc.themes.BOOTSTRAP]
        )
        self.app.config.suppress_callback_exceptions = True
        self.api = Api(self.server_app)

        # self.api.add_resource(self.Homepage(), "/demo")

        self.app.layout = dbc.Alert(
            "Artful Thinking."
        )

    def render_dash_apps(self):
        for conf_dapp in self.conf_render_dash_apps:
            if "active" in conf_dapp:
                if not conf_dapp["active"]:
                    continue

            dapp_layout_list = []
            for dapp_item in conf_dapp["dapp_list"]:
                if "active" in conf_dapp:
                    if not conf_dapp["active"]:
                        continue
                if dapp_item["name"] not in self.dash_apps:
                    print("異常：「" + dapp_item["name"] + "」DASH-APPがありません。")
                    continue

                dapp_layout_list.append( self.dash_apps[dapp_item["name"]].layout )
                dapp_layout_list.append(html.Br())

            dapp_layout = html.Div(
                    dapp_layout_list
                )

            self.update_dash_app_layout(conf_dapp["name"], dapp_layout)

    def update_dash_apps(self):
        for conf_dapp in self.conf_update_dash_apps:
            if "active" in conf_dapp:
                if not conf_dapp["active"]:
                    continue
            print("処理：" + conf_dapp["proc_conf"]["name"])
            getattr(self, conf_dapp["proc_conf"]["name"])(\
                conf_dapp["name"], conf_dapp["proc_conf"])

    def intfImgReco(self):
        try:
            return self.procImgReco()
        except Exception as e_msg:
            raise Exception("Failed to recognize the image") \
                from e_msg

    def procImgReco(self):

        if flask_request.method == 'POST':
            return "post"
        elif flask_request.method == 'GET':
            return "get"

    def Homepage(self):
        return html.Div()

    def get_dummy_image(self, height, width, pix_color=255):
        # with open('%s' %img_path, "rb") as image_file:
        #     encoded_string = base64.b64encode(image_file.read()).decode()
        return array_to_data_url(\
            pix_color * np.ones((height, width, 3), np.uint8))
        # return "data:image/png;base64," + encoded_string

    def img_to_str(self, img):
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        buffered.seek(0)
        img_byte = buffered.getvalue()
        return "data:image/png;base64," + base64.b64encode(img_byte).decode()

    def base64_to_img_buf(self, img_str):
        return Image.open(BytesIO(base64.b64decode(img_str[22:])))

    def update_dapp_layout_homepage(self) -> None:
        proc_conf = self.conf_dapp_homepage["proc_conf"]

        dapp_layout_list = []

        conf_style = {
                'flex-grow': 1,
                'min-height': '700px'
            }
        columns = ['type', 'width', 'height', 'scaleX', 'strokeWidth', 'path']

        elmid_annot_canvas = 'annot-canvas'
        # elmid_annot_canvas_table = 'annot-canvas-table'
        elmid_annot_image_0 = 'elmid-annot-image-0'
        elmid_annot_image_1 = 'elmid-annot-image-1'
        elmid_tile_img_pred = 'elmid-tile-image-pred'
        elmid_tess_row_cnt = 'elmid-tess-row-cnt'
        elmid_tess_col_cnt = 'elmid-tess-col-cnt'
        elmid_tess_img_sel = 'elmid-tess-image-selection'
        elmid_tess_agl_sel = 'elmid-tess-angle-selection'
        elmid_tess_gen_btn = 'elmid-tess-gen'
        elmid_tess_img = 'elmid-tess-img'

        params = [
        ]
        canvas_w = 500
        canvas_h = 500
        pred_thr = 0.3
        dapp_layout_list.append(\
            dbc.Card([
                dbc.CardHeader("Step 1: Draw a tessellation tile image on the canvas below."),
                dbc.CardBody(
                    style=conf_style,
                    children=[
                        html.Div([
                            dbc.Card([
                                dbc.CardBody(
                                    children=[
                                        html.H4("Digital Canvas", className="card-title"),
                                        DashCanvas(
                                            id=elmid_annot_canvas,
                                            lineWidth=3,
                                            lineColor='black',
                                            width=canvas_w,
                                            height=canvas_h,
                                            tool="pencil",
                                            filename=self.get_dummy_image(canvas_w, canvas_h),
                                            # self.get_dummy_image("data/cases/case_1/bkimg/FFFFFF-0.png"),
                                            hide_buttons=['line', 'pan', 'select', 'rectangle'],
                                            goButtonTitle='to Recognize the Drawing'
                                        )
                                    ],
                                    style={"textAlign":"center"},
                                    className="five columns"
                                )
                                ],
                                color="secondary", outline=True
                            ),
                            html.Hr(),
                            dbc.Row([
                                dbc.Col(
                                    dbc.Card([
                                        dbc.CardBody(
                                            children=[
                                                html.H4("Recognition (with ranked score)", className="card-title"),
                                                DataTable(
                                                    id=elmid_tile_img_pred,
                                                    columns=(
                                                        [{'id': 'label', 'name': 'Label that looks like'}] +
                                                        [{'id': 'score', 'name': 'Score ranking'}]
                                                        # [{'id': p, 'name': p} for p in params]
                                                    ),
                                                    data=[
                                                        # dict(Label=i, **{param: 0 for param in params})
                                                        # for i in range(0, 0)
                                                    ],
                                                    editable=True,
                                                    style_table={
                                                        # 'maxHeight': '50ex',
                                                        'overflowY': 'scroll',
                                                        'width': '100%',
                                                        'max-height': '500px'
                                                    },
                                                    style_header={
                                                        'fontWeight': 'bold',
                                                        'textAlign': "center",
                                                        'backgroundColor': 'white',
                                                    },
                                                    style_cell={
                                                        'fontFamily': 'Open Sans',
                                                        'textAlign': 'left',
                                                        'min-height': '25px',
                                                        'padding': '1px 22px',
                                                        'whiteSpace': 'inherit',
                                                        'overflow': 'hidden',
                                                        'textOverflow': 'ellipsis',
                                                    }
                                                )
                                            ],
                                            style={"textAlign": "center"}
                                        )
                                    ]),
                                xl=6),
                                dbc.Col(
                                    dbc.Card([
                                        dbc.CardBody(
                                            children=[
                                                html.H4(f"Advice (score >= {pred_thr})", className="card-title"),
                                                dbc.Row([
                                                    dbc.Col(
                                                        html.Div(
                                                            [
                                                                html.Img(
                                                                    id=elmid_annot_image_0,
                                                                    height=160,
                                                                    width=160,
                                                                    style={
                                                                        "border-color":"black",
                                                                        "border-style": "solid"
                                                                    },
                                                                    alt="No advice image here."
                                                                ),
                                                                html.H5("Advice 1")
                                                            ]
                                                        ),
                                                    sm=6),
                                                    dbc.Col(
                                                        html.Div(
                                                            [
                                                                html.Img(
                                                                    id=elmid_annot_image_1,
                                                                    height=160,
                                                                    width=160,
                                                                    # src=self.get_dummy_image(500, 500),
                                                                    style={
                                                                        "border-color":"black",
                                                                        "border-style": "solid"
                                                                    },
                                                                    alt="No advice image here."
                                                                ),
                                                                html.H5("Advice 2")
                                                            ]
                                                        ),
                                                    sm=6)
                                                ])
                                            ],
                                            style={"textAlign": "center"}
                                        )
                                        ]
                                    ),
                                xl=6)
                            ]),
                        ]),
                    ]
                )
            ],
            color="primary", outline=True)
        )

        # tess_cols = [f"Col_{i+1}" for i in range(0,3)]

        dapp_layout_list.append(dbc.Card([
            dbc.CardHeader("Step 2: Tuning the parameters to generate tessellation."),
            dbc.CardBody(
                style=conf_style,
                children=[
                    html.Div([
                        dbc.Card([
                            dbc.CardBody(
                                children=[
                                    html.H4("Scale of Tessellation (Amount of Columsn and Rows)", className="card-title"),
                                    dbc.Row(
                                        children=[
                                            dbc.Col(
                                                dcc.Dropdown(
                                                    id=elmid_tess_col_cnt,
                                                    placeholder="Select the amount of columns",
                                                    options=[
                                                        {'label': f'# Col = {i}', 'value': i} \
                                                            for i in range(2, 10)
                                                    ],
                                                    value=2,
                                                    multi=False
                                                ), sm=6
                                            ),
                                            dbc.Col(
                                                dcc.Dropdown(
                                                    id=elmid_tess_row_cnt,
                                                    placeholder="Select the amount of rows",
                                                    options=[
                                                        {'label': f'# Row = {i}', 'value': i} \
                                                            for i in range(2, 10)
                                                    ],
                                                    value=2,
                                                    multi=False
                                                ), sm=6
                                            )
                                        ]
                                    )
                                ],
                                style={"textAlign": "center"}
                            )
                        ]),
                        # DataTable(
                        #     id=elmid_annot_canvas_table,
                        #     style_cell={'textAlign': 'left'},
                        #     columns=[{"name": i, "id": i} for i in columns]),
                        # html.Br(),
                        html.Hr(),
                        # dcc.Graph(id='table-editing-simple-output')
                        dbc.Card([
                            dbc.CardBody(
                                children=[
                                    html.H4("Please fill a number in cell to indicate the adviced image for each tile.", className="card-title"),
                                    DataTable(
                                        id=elmid_tess_img_sel,
                                        columns=(
                                        ),
                                        data=[
                                        ],
                                        style_data_conditional=[],
                                        editable=True,
                                        style_table={
                                            # 'maxHeight': '50ex',
                                            'overflowY': 'scroll',
                                            'width': '100%',
                                            'max-height': '500px'
                                        },
                                        style_header={
                                            'fontWeight': 'bold',
                                            'textAlign': "center",
                                            'backgroundColor': 'white',
                                        },
                                        style_cell={
                                            'fontFamily': 'Open Sans',
                                            'textAlign': 'left',
                                            'min-height': '25px',
                                            'padding': '1px 22px',
                                            'whiteSpace': 'inherit',
                                            'overflow': 'hidden',
                                            'textOverflow': 'ellipsis',
                                        }
                                    )
                                ]
                            )
                        ]),
                        html.Br(),
                        dbc.Card([
                            dbc.CardBody(
                                children=[
                                    html.H4("Please fill a number in cell to indicate the image direction in each tile.", className="card-title"),
                                    DataTable(
                                        id=elmid_tess_agl_sel,
                                        columns=(
                                        ),
                                        data=[
                                        ],
                                        style_data_conditional=[],
                                        editable=True,
                                        style_table={
                                            # 'maxHeight': '50ex',
                                            'overflowY': 'scroll',
                                            'width': '100%',
                                            'max-height': '500px'
                                        },
                                        style_header={
                                            'fontWeight': 'bold',
                                            'textAlign': "center",
                                            'backgroundColor': 'white',
                                        },
                                        style_cell={
                                            'fontFamily': 'Open Sans',
                                            'textAlign': 'left',
                                            'min-height': '25px',
                                            'padding': '1px 22px',
                                            'whiteSpace': 'inherit',
                                            'overflow': 'hidden',
                                            'textOverflow': 'ellipsis',
                                        }
                                    )
                                ]
                            )
                        ]),
                        html.Hr(),
                        dbc.Card([
                            dbc.CardBody(
                                children=[
                                    html.Button('to Generate Tessellation',
                                        id=elmid_tess_gen_btn,
                                        n_clicks=0,
                                        style={"color": "blue"})
                                ],
                                style={"textAlign":"center"}
                            )
                            ]
                        )
                    ])
                ]
            )
            ],
            color="primary", outline=True
        ))

        info_card = dbc.Card([
                dbc.CardHeader("A Support System for Artful Design of Tessellations Drawing"),
                dbc.CardBody(
                    [
                        dbc.ListGroup(
                        [
                            dbc.ListGroupItem(info["text"]) for\
                                info in proc_conf["describe"]["info_list"]
                        ])
                    ]
                )
            ])
        self.app.layout =  html.Div([
                dbc.Row([
                    dbc.Col(html.Div([info_card]), lg=12)
                ], align="center"),
                html.Br(),
                dbc.Row([
                    dbc.Col(html.Div([x]), lg=12/len(dapp_layout_list)) \
                        for x in dapp_layout_list
                ], align="center"),
                html.Br(),
                dbc.Card([
                    dbc.CardHeader("Candidate of Tessellation"),
                    dbc.CardBody(
                        style=conf_style,
                        children=[
                            html.Div([
                                html.Img(
                                    id=elmid_tess_img,
                                    width="80%",
                                    style={
                                        "border-color":"black",
                                        "border-style": "solid"
                                    }
                                )
                            ]),
                        ]
                    )
                    ],
                    color="success", outline=True,
                    style={
                        "textAlign":"center"
                    }
                )
            ])

        @self.app.callback(
            [
                Output(component_id=elmid_tile_img_pred, component_property='data'),
                Output(component_id=elmid_annot_image_0, component_property='src'),
                Output(component_id=elmid_annot_image_0, component_property='alt'),
                Output(component_id=elmid_annot_image_1, component_property='src'),
                Output(component_id=elmid_annot_image_1, component_property='alt')
            ],
            [
                Input(component_id=elmid_annot_canvas, component_property='json_data')
            ]
        )
        def _update_img_proc(str_data):
            if str_data:
                mask = parse_jsonstring(str_data) #(canvas_w, canvas_h)
            else:
                raise PreventUpdate
            # return array_to_data_url(Image.fromarray(mask))
            try:
                mask_arry = (255 * mask).astype(np.uint8)
                mask_arry = np.where(\
                    mask_arry >= 255, 0, 255).astype(np.uint8)
                img_url = array_to_data_url(mask_arry)

                save_img_dp = os.path.join(self.case_dir_path, "bkimg", \
                    datetime.now().strftime("%Y-%m-%d"))
                os.makedirs(save_img_dp, exist_ok=True)
                self.base64_to_img_buf(img_url).save(os.path.join(save_img_dp, \
                    f'draw_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png'))

            except Exception as e_msg:
                raise Exception("Failed to get image url.") from e_msg

            try:
                # pred_rslt = self.cnnc_tile.pred_mask_array(mask_arry)
                img_post = {
                    "img_str": img_url
                }
                http_url = '{}://{}:{}{}'.format(\
                    self.conf_model_tile["interface"]['ptcl'],
                    self.conf_model_tile["interface"]['host'],
                    self.conf_model_tile["interface"]['port'],
                    self.conf_model_tile["interface"]["conf_endpoint"]\
                        ["pred"]["endpoint_api"]
                )
                self.logger.info(http_url)
                r = requests.post(
                    http_url,
                    data=img_post,
                    verify=False
                )
                rv = r.content
                if type(rv) == dict:
                    rv = namedtuple('Struct', rv.keys())(*rv.values())
                pred_rslt = to_json(json.loads(rv.decode())["result"])["prediction"]

                top_k = 2
                list_img_str = []
                # img_h = canvas_h
                # img_w = canvas_w
                for idx, itm in enumerate(pred_rslt):
                    if itm["score"] < pred_thr:
                        continue
                    if idx < top_k:
                        file_path = random.choice(glob(os.path.join(\
                            self.case_dir_path, "advice", itm["label"], "*.png")))
                        img = Image.open(file_path)
                        # img_h, img_w = img.size
                        list_img_str.append(self.img_to_str(img))

                advice_text = "No advice image here."
                if len(list_img_str)>0:
                    img_str_0 = list_img_str[0]
                    advice_txt_0 = "Advice 1 available."
                else:
                    img_str_0 = self.get_dummy_image(1, 1, 230)
                    #"data:image/png;base64"
                    advice_txt_0 = advice_text
                if len(list_img_str)>1:
                    img_str_1 = list_img_str[1]
                    advice_txt_1 = "Advice 2 available."
                else:
                    # img_str_1 = self.get_dummy_image(1, 1, 230)
                    img_str_1 = self.get_dummy_image(1, 1, 230)
                    #"data:image/png;base64"
                    advice_txt_1 = advice_text
                    # ,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII=
                print(pred_rslt)
            except Exception as e_msg:
                raise Exception("Failed to predict image.") from e_msg

            return pred_rslt, img_str_0, advice_txt_0, img_str_1, advice_txt_1

        @self.app.callback(
            [
                Output(component_id=elmid_tess_img_sel, component_property="columns"),
                Output(component_id=elmid_tess_img_sel, component_property="data"),
                Output(component_id=elmid_tess_img_sel, component_property="style_data_conditional"),
                Output(component_id=elmid_tess_agl_sel, component_property="columns"),
                Output(component_id=elmid_tess_agl_sel, component_property="data"),
                Output(component_id=elmid_tess_agl_sel, component_property="style_data_conditional")
            ],
            [
                Input(component_id=elmid_tess_col_cnt, component_property='value'),
                Input(component_id=elmid_tess_row_cnt, component_property='value')
            ],
            [
                State(component_id=elmid_tess_img_sel, component_property="data"),
                State(component_id=elmid_tess_agl_sel, component_property="data"),
                State(component_id=elmid_annot_image_0, component_property="alt"),
                State(component_id=elmid_annot_image_1, component_property="alt")
            ]
        )
        def _init_tess_param_table(
                col_cnt,
                row_cnt,
                data_tess_img_sel,
                data_tess_agl_sel,
                alt_txt_advice_0,
                alt_txt_advice_1
            ):

            lst_img_sel = []
            for row_val in data_tess_img_sel:
                lst_img_sel.append([row_val[x] for x in row_val.keys()])
            lst_agl_sel = []
            for row_val in data_tess_agl_sel:
                lst_agl_sel.append([row_val[x] for x in row_val.keys()])

            def get_val(list_val, i, j):
                if (i > 0) and (i <len(list_val)):
                    if (j > 0) and (j <len(list_val[i])):
                        return list_val[i][j]

                return 1

            columns_img_sel = [{'id': "Rows", 'name': ''}] +\
                [{'id': f"Col_{i}", 'name': f"ColVal_{i+1}"} for i in range(0,col_cnt)]
            columns_agl_sel = [{'id': "Rows", 'name': ''}] +\
                [{'id': f"Col_{i}", 'name': f"ColVal_{i+1}"} for i in range(0,col_cnt)]

            self.logger.info(columns_img_sel)

            data_tess_img_sel = [
                dict(Rows=f"RowVal_{i+1}", **{f"Col_{j}": get_val(lst_img_sel, i, j) for j in range(0,col_cnt)})
                for i in range(0, row_cnt)
            ]
            data_tess_agl_sel = [
                dict(Rows=f"RowVal_{i+1}", **{f"Col_{j}": get_val(lst_agl_sel, i, j) for j in range(0,col_cnt)})
                for i in range(0, row_cnt)
            ]
            lst_img_idx_avb = []
            if any(char.isdigit() for char in alt_txt_advice_0):
                lst_img_idx_avb.append("0")
            if any(char.isdigit() for char in alt_txt_advice_1):
                lst_img_idx_avb.append("1")
            style_img_sel = [
                {
                    "if": {
                        'column_id': f'Rows'
                    },
                    'column_editable': False,
                    'backgroundColor': 'rgb(240, 240, 240)'
                }
            ]
            # + \
            # [
            #     {
            #         "if": {
            #             'column_id': f'Col_{i+1}',
            #             'filter_query': '{Col_' + str(i+1) + '} = 1'
            #         },
            #         'backgroundColor': 'rgb(240, 240, 240)'
            #     } for i in range(0,col_cnt)
            # ]
            style_agl_sel = [
                {
                    "if": {
                        'column_id': f'Rows'
                    },
                    'column_editable': False,
                    'backgroundColor': 'rgb(240, 240, 240)'
                }
            ]
            return columns_img_sel, data_tess_img_sel, style_img_sel,\
                columns_agl_sel, data_tess_agl_sel, style_agl_sel

        @self.app.callback(
            Output(component_id=elmid_tess_img, component_property='src'),
            [
                Input(component_id=elmid_tess_gen_btn, component_property='n_clicks')
            ],
            [
                State(component_id=elmid_tess_img_sel, component_property="data"),
                State(component_id=elmid_tess_agl_sel, component_property="data"),
                State(component_id=elmid_annot_image_0, component_property="alt"),
                State(component_id=elmid_annot_image_1, component_property="alt"),
                State(component_id=elmid_annot_image_0, component_property='src'),
                State(component_id=elmid_annot_image_1, component_property='src'),
            ]
        )
        def _generate_tessellation(
                tess_gen_btn_clk,
                data_tess_img_sel,
                data_tess_agl_sel,
                alt_txt_advice_0,
                alt_txt_advice_1,
                img_str_0,
                img_str_1
            ):
            # if (tess_gen_btn_clk > 0):
            self.logger.info("Clicked")
            lst_img_idx_avb = []
            lst_img_avb = []
            if any(char.isdigit() for char in alt_txt_advice_0):
                lst_img_idx_avb.append(0)
                lst_img_avb.append(img_str_0)
            if any(char.isdigit() for char in alt_txt_advice_1):
                lst_img_idx_avb.append(1)
                lst_img_avb.append(img_str_1)

            row_cnt = len(data_tess_img_sel)
            if len(data_tess_img_sel)>0:
                col_cnt = len(data_tess_img_sel[0])-1 # excluding row index
            else:
                col_cnt = 0

            tess_param = []

            delta_agl=-7
            for idx_i in range(0, row_cnt):
                tp_row = []
                for idx_j in range(0, col_cnt):
                    try:
                        img_idx = int(data_tess_img_sel[idx_i][f"Col_{idx_j}"])-1
                    except:
                        img_idx = -1
                    try:
                        agl_idx = int(data_tess_agl_sel[idx_i][f"Col_{idx_j}"])
                    except:
                        agl_idx = 0
                    tp_row.append(
                        {
                            "img_idx": img_idx if img_idx in lst_img_idx_avb else -1,
                            "img_agl": agl_idx*90+delta_agl
                        }
                    )
                tess_param.append(tp_row)
            self.logger.info(json.dumps(tess_param, indent=4))

            tess_conf = {
                "panel": "square",
                "tiles": tess_param
            }

            tess_post = {
                "tess_conf": to_json_str(tess_conf),
                "list_img_str": to_json_str(lst_img_avb)
            }

            http_url = '{}://{}:{}{}'.format(\
                self.conf_model_tile["interface"]['ptcl'],
                self.conf_model_tile["interface"]['host'],
                self.conf_model_tile["interface"]['port'],
                self.conf_model_tile["interface"]["conf_endpoint"]\
                    ["tessellate"]["endpoint_api"]
            )
            self.logger.info(http_url)
            r = requests.post(
                http_url,
                data=tess_post,
                verify=False
            )
            rv = r.content
            if type(rv) == dict:
                rv = namedtuple('Struct', rv.keys())(*rv.values())
            tess_rslt = {}
            try:
                tess_rslt = to_json(json.loads(rv.decode())["result"])["tessellation"]

                save_img_dp = os.path.join(self.case_dir_path, "bkimg", \
                    datetime.now().strftime("%Y-%m-%d"))
                os.makedirs(save_img_dp, exist_ok=True)
                self.base64_to_img_buf(tess_rslt).save(os.path.join(save_img_dp, \
                    f'tess_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png'))
            except:
                self.logger.warning("Failed to decode the prediction result.")
            return tess_rslt
    def update_dash_app_layout(self, dapp_name, dapp_layout):

        self.dash_apps[dapp_name] = Dash(
            dapp_name,
            server=self.server_app,
            url_base_pathname=self.rendering_server["path"] + "/" + dapp_name + "/",
            external_stylesheets=[dbc.themes.BOOTSTRAP]
        )

        self.dash_apps[dapp_name].config.suppress_callback_exceptions = True

        self.dash_apps[dapp_name].layout = dapp_layout
        return self.dash_apps[dapp_name].server

    def run(self):

        self.update_dapp_layout_homepage()

        self.update_dash_apps()
        self.render_dash_apps()

        self.server_app.run(
            host=self.rendering_server["host"],
            port=self.rendering_server["port"],
            debug=self.rendering_server["debug"],
            threaded=True
        )

        # self.app.run_server(
        # self.dash_apps["DAPP: Canvas"].run_server(
        #     host=self.rendering_server["host"],
        #     port=self.rendering_server["port"],
        #     debug=self.rendering_server["debug"]
        # )

def main(sys_argv):
    mfa = MainFuncAssistance()
    cf_logger = mfa.get_logger("TessellationInterface")
    cf_logger.info("プローグラムを起動しました。")

    task_conf = {}
    conf_file_path_list = [
        'conf/task_conf.json'
    ]
    for conf_file_path in conf_file_path_list:
        with open(conf_file_path, encoding='utf-8') as fj:
            task_conf = dict(mfa.mergedicts(task_conf, json.load(fj)))
    if len(sys_argv) > 1:
        task_conf["conf_case"]["case_dir_path"] = f"data/cases/{sys_argv[1]}"
    case_dir_path = task_conf["conf_case"]["case_dir_path"]
    with open('{}/conf/task_conf.json'.format(case_dir_path), encoding='utf-8') as fj:
        task_conf = dict(mfa.mergedicts(task_conf, json.load(fj)))

    tsr = TessellationRendering(task_conf, cf_logger)
    tsr.run()

if __name__ == "__main__":
    main(sys.argv)