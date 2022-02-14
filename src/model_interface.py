"""model_interface.py: The model interface for Tessellation Drawing."""

__author__ = "Luyi HUAANG <luyi.lancle.huaang@gmail.com>"
__copyright__ = "Copyright 2020"

from nnabla.utils import nnp_graph
from PIL import Image, ImageChops
from io import BytesIO
import base64
import numpy as np
from os.path import join as os_path_join
from flask import Flask, request as flask_request, jsonify
from dash import Dash
import dash_bootstrap_components as dbc
from flask_restful import Resource, Api
from endpoint_action import EndpointAction
import json
from assist import to_json_str, to_json


class ImageTransformation:
    def __init__(self):
        self.img_sqr_size = 256

    def get_bg_color(self, img_smp):
        return img_smp.getpixel((0, 0))

    def to_monochrome(self, img_smp):
        return img_smp.convert("L")

    def crop(self, im, bg_color=(255, 255, 255)):
        bg = Image.new(im.mode, im.size, bg_color)
        diff = ImageChops.difference(im, bg)
        bbox = diff.getbbox()
        if bbox:
            return im.crop(bbox)

    def bg_color_to_transparency(self, img_smp):
        x = np.asarray(img_smp.convert("RGBA")).copy()

        x[:, :, 3] = (255 * (x[:, :, :3] != 255).any(axis=2)).astype(np.uint8)

        return Image.fromarray(x)

    def resize_tile_img(self, img_smp, img_sqr_size=None):
        if img_sqr_size == None:
            img_sqr_size = self.img_sqr_size
        return img_smp.resize((img_sqr_size, img_sqr_size), Image.LANCZOS)

    def tessellationGeneration(self, img_list, tess_param):
        assert type(img_list) == list, "Type of image list must be list."
        if (len(img_list) <= 0) or (len(tess_param) <= 0):
            return Image.new("RGB", (1, 1), (255, 255, 255))
        assert len(img_list) > 0, "Image list len shall be at least 1."
        assert len(tess_param) > 0, "Tessellation param row len shall be at least 1."
        assert len(tess_param[0]) > 0, "Tessellation param col len shall be at least 1."

        img_list_rs = []
        img_sqr_size = 200
        for img in img_list:
            img_list_rs.append(img.resize((img_sqr_size, img_sqr_size), Image.LANCZOS))
        tile_h, tile_w = img_list_rs[0].size
        tile_h_margin = 47
        tile_w_margin = 47
        tess_w = (
            len(tess_param) * (tile_h - int(np.ceil(tile_h_margin))) + tile_w_margin
        )
        tess_h = (
            len(tess_param[0]) * (tile_w - int(np.ceil(tile_w_margin))) + tile_h_margin
        )
        img_bg = Image.new("RGB", (tess_h, tess_w), (255, 255, 255))
        for idx_r, tp_r in enumerate(tess_param):
            tile_w_offset = (tile_w - tile_w_margin) * idx_r
            for idx_c, tp_c in enumerate(tp_r):
                tile_h_offset = (tile_h - tile_h_margin) * idx_c
                if (tp_c["img_idx"] < 0) or (tp_c["img_idx"] >= len(img_list)):
                    continue
                img_tile = (
                    img_list_rs[tp_c["img_idx"]].copy().rotate(tp_c["img_agl"] - 90)
                )
                img_bg.paste(img_tile, (tile_h_offset, tile_w_offset), img_tile)
        return img_bg

    def img_to_str(self, img):
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        buffered.seek(0)
        img_byte = buffered.getvalue()
        return "data:image/png;base64," + base64.b64encode(img_byte).decode()


class ModelUtilityNNC:
    def __init__(self, conf_model, case_dir_path, model_selected="model_tile_reco"):

        assert "models" in conf_model, f'Key "models" in not in config'
        assert model_selected in conf_model, f'Key "{model_selected}" in not in config'
        # assert "case_dir_path" in conf_model, \
        #     "Key \"case_dir_path\" in not in config"
        assert (
            "key" in conf_model[model_selected]
        ), f'Key "name" in not in selected model config'
        assert (
            conf_model[model_selected]["key"] in conf_model["models"]
        ), f'Model config "{conf_model[model_selected]}" is missing.'
        self.ini_model(
            conf_model["models"][conf_model[model_selected]["key"]], case_dir_path
        )
        self.img_trans = ImageTransformation()

    def ini_model(self, conf_model_selected, case_dir_path):
        # assert "selected" in conf_model, \
        #     "Key \"selected\" in not in model config"
        # assert conf_model["selected"] in conf_model, \
        #     f"Key \"{conf_model['selected']}\" in not in model onfig"

        mdl_key = ["path", "label"]
        assert all(
            x in conf_model_selected for x in mdl_key
        ), f'Key "{mdl_key}" in not in the selected model onfig'
        self.list_lbl = conf_model_selected["label"]
        try:
            self.imp_model(os_path_join(case_dir_path, conf_model_selected["path"]))
        except Exception as e_msg:
            raise Exception("Failed to import model") from e_msg

    def imp_model(self, model_path):
        self.nnp = nnp_graph.NnpLoader(model_path)
        self.graph = self.nnp.get_network("MainRuntime", batch_size=1)
        # Input variable name
        var_input = list(self.graph.inputs.keys())[0]
        # Output variable name
        var_output = list(self.graph.outputs.keys())[0]
        # Get input and output
        self.x = self.graph.inputs[var_input]
        self.y = self.graph.outputs[var_output]

    def use_model(self, img_path):
        # Open image
        img = Image.open(img_path)
        print(img_path)
        # Transform
        self.x.d = np.array(img) * (1.0 / 255.0)
        # Forward
        self.y.forward(clear_buffer=True)
        idx = np.argmax(self.y.d)
        print(self.list_lbl[idx] if idx >= 0 else None)
        return np.argsort(self.y.d)

    def get_labels_by_idx(self, idx):
        # assert type(idx) == int, "Index is not int."
        if type(idx) == int:
            assert (idx >= 0) and (
                idx < len(self.list_lbl)
            ), f"Index {idx} exeeds the range."
            return self.list_lbl[idx]
        elif type(idx) == list:
            assert all(
                [(x >= 0) and (x < len(self.list_lbl)) for x in idx]
            ), f"Index list {idx} exceed the range"
            return [self.list_lbl[x] for x in idx]
        else:
            raise TypeError(
                f"Type of index {idx} is {type(idx)}, " + "which is unsupported."
            )

    def pred_mask_array(self, mask_arry):
        pil_img = Image.fromarray(mask_arry).convert("RGB")
        # pil_img = self.img_tool.crop(pil_img)
        pil_img = self.img_trans.to_monochrome(
            self.img_trans.bg_color_to_transparency(
                self.img_trans.resize_tile_img(
                    # self.img_trans.crop(pil_img),
                    pil_img,
                    28,
                )
            )
        )

        pil_img.save("data/tmp.png")
        # orig_pred_result = self.use_model("data/tmp.png")[0]

        self.x.d = np.array(pil_img) * (1.0 / 255.0)
        # Forward
        self.y.forward(clear_buffer=True)
        orig_pred_result = np.argsort(self.y.d[0])

        ranked_score_idx = list(reversed(orig_pred_result))

        print(self.y.d[0])
        print(self.list_lbl)
        ranked_label_idx = self.get_labels_by_idx(ranked_score_idx)
        pred_rslt = []
        for idx, lbl in enumerate(ranked_label_idx):
            pred_rslt.append(
                {"label": lbl, "score": self.y.d[0][ranked_score_idx[idx]]}
            )

        return pred_rslt

    def pred_img_str(self, img_str):
        # pil_img = Image.fromarray(mask_arry).convert("RGB")
        # pil_img = self.img_tool.crop(pil_img)
        pil_img = Image.open(BytesIO(base64.b64decode(img_str[22:])))
        pil_img = self.img_trans.to_monochrome(
            self.img_trans.bg_color_to_transparency(
                self.img_trans.resize_tile_img(
                    # self.img_trans.crop(pil_img),
                    pil_img,
                    28,
                )
            )
        )

        pil_img.save("data/tmp.png")
        # orig_pred_result = self.use_model("data/tmp.png")[0]

        self.x.d = np.array(pil_img) * (1.0 / 255.0)
        # Forward
        self.y.forward(clear_buffer=True)
        orig_pred_result = np.argsort(self.y.d[0])

        ranked_score_idx = list(reversed(orig_pred_result))

        print(self.y.d[0])
        print(self.list_lbl)
        ranked_label_idx = self.get_labels_by_idx(ranked_score_idx)
        pred_rslt = []
        for idx, lbl in enumerate(ranked_label_idx):
            pred_rslt.append(
                {"label": lbl, "score": self.y.d[0][ranked_score_idx[idx]]}
            )

        return pred_rslt


class ModelInterface:
    def __init__(self, conf_case, model_inst, cf_logger):
        self.logger = cf_logger
        self.validate_conf_case(conf_case, model_inst)
        self.conf_model = conf_case["conf_model"]
        self.case_dir_path = conf_case["case_dir_path"]
        self.conf_model_inst = conf_case["conf_model"][model_inst]
        self.initialize_model(model_inst)
        self.initialize_interface(model_inst)

    def add_all_endpoints(self, conf_endpoint):
        # Add root endpoint
        for _key in conf_endpoint:
            self.add_endpoint(
                endpoint=conf_endpoint[_key]["endpoint"],
                endpoint_name=conf_endpoint[_key]["endpoint_api"],
                handler=getattr(self, conf_endpoint[_key]["handler"]),
            )

    def add_endpoint(self, endpoint=None, endpoint_name=None, handler=None):
        self.server_app.add_url_rule(
            endpoint, endpoint_name, EndpointAction(handler), methods=["POST", "GET"]
        )
        print(f"Added url rule: {endpoint}, {endpoint_name}, {handler}")
        # You can also add options here : "... , methods=['POST'], ... "

    def validate_conf_case(self, conf_case, model_inst):
        assert (
            "case_dir_path" in conf_case
        ), 'Key "case_dir_path" is missing in case config.'
        assert "conf_model" in conf_case, f'Key "conf_model" in not in config'
        assert (
            model_inst in conf_case["conf_model"]
        ), f'Key "{model_inst}" in not in config'
        assert (
            "interface" in conf_case["conf_model"][model_inst]
        ), f'Key "interface" in not in selected model config'

    def initialize_interface(self, model_inst):
        self.server_app = Flask(__name__)
        self.dash_apps = {}
        self.app = Dash(
            server=self.server_app,
            url_base_pathname=self.conf_model[model_inst]["interface"]["base"],
            external_stylesheets=[dbc.themes.BOOTSTRAP],
        )
        # self.app.config.suppress_callback_exceptions = True
        self.api = Api(self.server_app)
        # self.api.add_resource(self.Homepage(), "/demo")
        self.app.layout = dbc.Alert("Model Interface.")
        self.add_all_endpoints(
            self.conf_model[model_inst]["interface"]["conf_endpoint"]
        )

    def run(self):
        self.server_app.run(
            host=self.conf_model_inst["interface"]["host"],
            port=self.conf_model_inst["interface"]["port"],
            debug=self.conf_model_inst["interface"]["debug"],
            threaded=self.conf_model_inst["interface"]["threaded"],
        )

    def initialize_model(self, model_selected):
        self.model_nnc = ModelUtilityNNC(
            self.conf_model, self.case_dir_path, model_selected
        )

    def proc_pred(self):
        self.logger.info("Prediction")
        self.logger.info(flask_request.form.keys())
        result = {}
        if flask_request.method == "POST":
            # check data and config
            if flask_request.form.get("img_str"):
                img_str = flask_request.form["img_str"]
                pred_rslt = self.model_nnc.pred_img_str(img_str)
                result = {"result": to_json_str({"prediction": pred_rslt}).decode()}
                self.logger.info(type(result))
        elif flask_request.method == "GET":
            self.logger.warning("Please use post method.")
        else:
            raise Exception('Method "GET" is not supported.')
        return json.dumps(result)

    def proc_tess(self):
        self.logger.info("Tessellation Generation")
        self.logger.info(flask_request.form.keys())
        result = {}
        if flask_request.method == "POST":
            # check data and config
            if flask_request.form.get("list_img_str") and flask_request.form.get(
                "tess_conf"
            ):
                list_img_str = to_json(flask_request.form["list_img_str"])
                # self.logger.info(list_img_str)
                tess_conf = to_json(flask_request.form["tess_conf"])
                list_img = []
                for img_str in list_img_str:
                    list_img.append(Image.open(BytesIO(base64.b64decode(img_str[22:]))))
                self.logger.info(json.dumps(tess_conf["tiles"], indent=4))
                tess_img = self.model_nnc.img_trans.tessellationGeneration(
                    list_img, tess_conf["tiles"]
                )
                result = {
                    "result": to_json_str(
                        {"tessellation": self.model_nnc.img_trans.img_to_str(tess_img)}
                    ).decode()
                }
                # self.logger.info(type(result))
        elif flask_request.method == "GET":
            self.logger.warning("Please use post method.")
        else:
            raise Exception('Method "GET" is not supported.')
        return json.dumps(result)


def run_tessellation_generation(conf_case):
    img_smp_fp = "data/cases/case_3/advice/bird/3.png"
    list_img_fp = [img_smp_fp]
    list_img = [Image.open(x).resize((200, 200), Image.LANCZOS) for x in list_img_fp]
    delta_agl = -7
    tess_param = [
        [
            {"img_idx": 0, "img_agl": 90 + delta_agl},
            {"img_idx": 0, "img_agl": 180 + delta_agl},
        ],
        [
            {"img_idx": 0, "img_agl": 270 + delta_agl},
            {"img_idx": 0, "img_agl": 0 + delta_agl},
        ],
    ]
    tess_conf = {"panel": "square", "tiles": tess_param}

    img_rslt = ImageTransformation().tessellationGeneration(list_img, tess_conf)
    img_rslt.save("data/tmp.png")


def run_smp_model_nnc(conf_case):
    model_utility = ModelUtilityNNC(conf_case["conf_model"], conf_case["case_dir_path"])
    pred_imgs = [
        "../classification/data/cases/case_3/cure/fmt/png/bird/1.png",
        "../classification/data/cases/case_3/cure/fmt/png/fish/1.png",
        "../classification/data/cases/case_3/cure/fmt/png/face/1.png",
        "data/tmp.png",
    ]
    for pi in pred_imgs:
        model_utility.use_model(pi)


def get_conf_case():
    from assist import MainFuncAssistance
    import sys

    mfa = MainFuncAssistance()
    cf_logger = mfa.get_logger("ModelInterface")
    cf_logger.info("プローグラムを起動しました。")

    task_conf = {}
    conf_file_path_list = ["conf/task_conf.json"]
    for conf_file_path in conf_file_path_list:
        with open(conf_file_path, encoding="utf-8") as fj:
            task_conf = dict(mfa.mergedicts(task_conf, json.load(fj)))
    assert "conf_case" in task_conf, 'Key "conf_case" is not in config.'
    if len(sys.argv) > 1:
        assert (
            "case_dir_path" in task_conf["conf_case"]
        ), 'Key "case_dir_path" is not in case config.'
        task_conf["conf_case"]["case_dir_path"] = f"data/cases/{sys.argv[1]}"
    case_dir_path = task_conf["conf_case"]["case_dir_path"]
    with open("{}/conf/task_conf.json".format(case_dir_path), encoding="utf-8") as fj:
        task_conf = dict(mfa.mergedicts(task_conf, json.load(fj)))
    return task_conf["conf_case"], cf_logger


def main():

    conf_case, cf_logger = get_conf_case()

    # run_smp_model_nnc(conf_case)
    # run_tessellation_generation(conf_case)
    ms = ModelInterface(conf_case, "model_tile_reco", cf_logger)
    ms.run()


if __name__ == "__main__":
    main()
