"""
Chinese Retrieval tasks module.
Each task is defined in its own file with its associated loader.
"""

from .data_loaders.advertisegen import AdvertiseGen
from .data_loaders.ccovidnews import CCOVIDNews
from .data_loaders.chatmeddataset import ChatMedDataset
from .data_loaders.chef import CHEF
from .data_loaders.cmedqav2 import CMedQAV2
from .data_loaders.cmrc2018 import CMRC2018
from .data_loaders.csl import CSL
from .data_loaders.drcd import DRCD
from .data_loaders.dureader import DuReader
from .data_loaders.dureaderchecklist import DuReaderchecklist
from .data_loaders.lawgpt import LawGPT
from .data_loaders.lawzhidao import LawZhidao
from .data_loaders.lcsts import LCSTS
from .data_loaders.limazh import LIMAZH
from .data_loaders.mmarcozh import MMarcoZh
from .data_loaders.multicpr import MultiCPR
from .data_loaders.pawsxzh import PAWSXZh
from .data_loaders.refgpt import RefGPT
from .data_loaders.retrievaldatallm import RetrievalDataLLM
from .data_loaders.t2ranking import T2Ranking
from .data_loaders.thucnews import THUCNews
from .data_loaders.umetripqa import UMETRIPQA
from .data_loaders.webcpm import WebCPM
from .data_loaders.webqa import WebQA

__all__ = [
    "AdvertiseGen",
    "CCOVIDNews",
    "CHEF",
    "ChatMedDataset",
    "CMedQAV2",
    "CMRC2018",
    "CSL",
    "DRCD",
    "DuReader",
    "DuReaderchecklist",
    "LawGPT",
    "LawZhidao",
    "LCSTS",
    "LIMAZH",
    "MMarcoZh",
    "MultiCPR",
    "PAWSXZh",
    "RefGPT",
    "RetrievalDataLLM",
    "T2Ranking",
    "THUCNews",
    "UMETRIPQA",
    "WebCPM",
    "WebQA",
]
