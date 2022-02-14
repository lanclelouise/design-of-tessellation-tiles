"""assist.py: The assistant function for CLI operation."""

__author__ = "Luyi HUAANG <luyi.lancle.huang@gmail.com>"
__copyright__ = "Copyright 2020"

import logging
from logging.handlers import TimedRotatingFileHandler
import yaml
import readline
import json
import argparse
import sys
import os
from datetime import datetime
from csv import writer


class MainFuncAssistance:
    """Main function assistant tool (by Y. WU)
    (1) to capture the sys options
    (2) to initialize logger instance
    (3) to merge dict content
    """

    def __init__(
        self, logs_dir_path: str = "data/logs", char_encoding_default: str = "UTF-8"
    ):
        self.LOGS_DIR_PATH = logs_dir_path
        self.CHAR_ENCODING_DEFAULT = char_encoding_default
        os.makedirs(self.LOGS_DIR_PATH, exist_ok=True)

        self.FORMATTER = logging.Formatter(
            "%(asctime)s — %(name)s — %(levelname)s — %(message)s"
        )

    def get_console_handler(self):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.FORMATTER)
        return console_handler

    def get_file_handler(self):
        file_handler = TimedRotatingFileHandler(
            self.LOG_FILE, when="midnight", encoding=self.CHAR_ENCODING_DEFAULT
        )
        file_handler.setFormatter(self.FORMATTER)
        return file_handler

    def get_logger_casebase(self, logger_name: str):
        pass

    def get_logger(self, logger_name: str):

        now = datetime.now()  # current date and time
        dt_str = now.strftime("%Y-%m-%d")

        self.LOG_FILE = f"{self.LOGS_DIR_PATH}/{logger_name}_{dt_str}.log"

        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)  # better to have too much log than not enough
        logger.addHandler(self.get_console_handler())
        logger.addHandler(self.get_file_handler())
        # with this pattern, it's rarely necessary to propagate the error up to parent
        logger.propagate = False
        return logger

    def mergedicts(self, dict1: dict, dict2: dict):
        for k in set(dict1.keys()).union(dict2.keys()):
            if k in dict1 and k in dict2:
                if isinstance(dict1[k], dict) and isinstance(dict2[k], dict):
                    yield (k, dict(self.mergedicts(dict1[k], dict2[k])))
                else:
                    # If one of the values is not a dict, you can't continue merging it.
                    # Value from second dict overrides one in first and we move on.
                    yield (k, dict2[k])
                    # Alternatively, replace this with exception raiser to
                    # alert you of value conflicts.
                    # however, here we only merge the dict-type content,
                    # but does not merge other types of content (e.g.: list).
            elif k in dict1:
                yield (k, dict1[k])
            else:
                yield (k, dict2[k])

    def getJsonObjByPath(self, path_in_json: str, json_data: dict = {}):
        jp_keys = path_in_json.split(".")
        jv_data = json_data
        for _key in jp_keys:
            if _key == "":
                continue
            # if hasattr(jv_data, "keys"):
            #     continue
            if _key in jv_data.keys():
                jv_data = jv_data[_key]
            else:
                jv_data = None
        return jv_data

    def getOptions(self, args, args_conf: list):
        parser = argparse.ArgumentParser(description="Parses command.")
        try:
            for ac_key in args_conf.keys():
                if "value" in args_conf[ac_key]:
                    parser.add_argument(
                        f"-{args_conf[ac_key]['opt_key']}",
                        f"--{ac_key}",
                        help=args_conf[ac_key]["help"],
                    )
                else:
                    parser.add_argument(
                        f"-{args_conf[ac_key]['opt_key']}",
                        f"--{ac_key}",
                        action="store_true",
                        help=args_conf[ac_key]["help"],
                    )
        except Exception as e:
            raise Exception(f"Failed to parse args by referring to {args_conf}") from e
        # parser.add_argument("-c", "--cases_idx", help="Your test case index.")
        # parser.add_argument("-p", "--pkg_label", help="Your package label.")
        # parser.add_argument("-r", "--role_label", help=\
        #     "The testing role: client (send req.) or host (receive req.).")
        # parser.add_argument("-d", "--sdlc_label", help=\
        #     "The SDLC (software develoment life cycle) label: "
        #     "PT (Program Test), IT (Integration Test), ST (System Test).")
        options = parser.parse_args(args)
        return options

    def sync_json_yaml_file(
        self,
        jy_file_path: str,  # path of of json/yaml file
        # only json/yaml-formated content
        # is allowed
        in_encoding: str = "utf-8",  # encoding of the input file,
        # default is utf-8
        out_encoding: str = "utf-8"  # encoding of the output file,
        # default is utf-8
    ):
        """to sync JSON and YAML file content
        required package: yaml, os, json
        """

        assert os.path.exists(jy_file_path), f"File does not exis: {jy_file_path}"

        str_file_name, str_ext_name = os.path.split(jy_file_path)

        with open(jy_file_path, "r", encoding=in_encoding) as fjy:
            try:
                if str_ext_name.lower() == ".json":
                    jy_data = json.load(fjy)
                    with open(f"{str_file_name}.yaml", "w", encoding="utf-8") as fy:
                        yaml.dump(jy_data, fy, allow_unicode=True)
                elif str_ext_name.lower() == ".yaml":
                    with open(f"{str_file_name}.json", "w", encoding="utf-8") as fj:
                        json.dump(jy_data, fj, indent=4, ensure_ascii=False)
            except Exception as e:
                raise f"Failed to exchange json/yaml content." from e

    def append_list_as_row(self, file_name, list_of_elem):
        # Open file in append mode
        with open(file_name, "a+", newline="") as write_obj:
            # Create a writer object from csv module
            csv_writer = writer(write_obj)
            # Add contents of list as last row in the csv file
            csv_writer.writerow(list_of_elem)

    def rlinput(self, prompt, prefill=""):
        """input via readline"""
        readline.set_startup_hook(lambda: readline.insert_text(prefill))
        try:
            return input(prompt)  # or raw_input in Python 2
        finally:
            readline.set_startup_hook()


import pandas as pd
from io import StringIO
import base64, pickle, zlib


def to_csv_str(data):
    # encode of pandas data
    # utf8_data = to_utf8(data)
    str_data = base64.b64encode(zlib.compress(pickle.dumps(data)))
    return str_data


def to_json_str(data):
    # encode of json data
    str_data = base64.b64encode(zlib.compress(pickle.dumps(data)))
    return str_data


def to_pd(data):
    # decode of pandas data
    pd_data = pickle.loads(zlib.decompress(base64.b64decode(data)))
    return pd_data


def to_json(data):
    # decode of json data
    json_data = pickle.loads(zlib.decompress(base64.b64decode(data)))
    return json_data


def to_utf8(data):
    # convert pandas dataframe to utf8
    tmp_data = data.to_csv(encoding="utf-8")
    tmp_data = StringIO(tmp_data)
    utf8_data = pd.read_csv(tmp_data, sep=",", index_col=0, encoding="utf-8")
    return utf8_data


def time_text(dict, msg):
    return f"{dict['task']}!{dict['status']}!{dict['allStep']}!{dict['nowStep']}!{msg}"


def data_step_analysis(config):
    return len(config["wl_data_creation"])


def to_model_str(model):
    # encode of model
    str_model = base64.b64encode(zlib.compress(pickle.dumps(model)))
    return str_model


def to_model(data):
    # decode of model
    model_data = pickle.loads(zlib.decompress(base64.b64decode(data)))
    return model_data
