import numpy
import os

from bluesky_tiled_plugins import TiledWriter
from bluesky.callbacks.buffer import BufferingWrapper
from tiled.client import from_uri


import bson
import copy
import os.path
import re
import uuid
from pathlib import Path

from bluesky_tiled_plugins.writing.tiled_writer import RunNormalizer

# In theory the columns can vary (set in the Resource document).
pb9_columns = ("ts_s", "ts_ns", "encoder", "index", "state")
pb9_dtype_list = [(name, "<i8") for name in pb9_columns]
pb9_dtype = pb1_dtype = pb4_dtype = numpy.dtype(pb9_dtype_list)

apb_columns = ("timestamp", "i0", "it", "ir", "iff", "aux1", "aux2", "aux3", "aux4")
apb_dtype_list = [("timestamp", "<f8")] + [(name, "<i4") for name in apb_columns[1:]]
apb_dtype = numpy.dtype(apb_dtype_list)

# PizzaBox ADC file for spec "PIZZABOX_AN_FILE_TXT" and "PIZZABOX_AN_FILE_TXT_PD"
pba_dtype_list = [("timestamp", "<i8"), ("col1", "<i8"), ("col2", "<i8"), ("col3", "<U10")]
pba_dtype = numpy.dtype(pba_dtype_list)
pba_dkey_pattern = r'^pba\d+_adc\d+$'   # e.g. 'pba1_adc7'

apb_trigger_dtype = numpy.dtype([("timestamp", "<f8"), ("transition", "<i4")])
LENGTH = 100_000
APB_AVE_FILENAMES = {"apb_ave_filename_bin", "apb_ave_filename_txt"}

# Mapping from spec to mimetype for use in TiledWriter
# TODO: Only keep the necessary specs here
MIMETYPE_LOOKUP = {
        "hdf5": "application/x-hdf5",
        "A1_HDF5": "application/x-hdf5",  # esm_patches:A1SoftFileHandler(HDF5DatasetSliceHandler)
        "AD_CBF": "multipart/related;type=image/tiff",
        "AD_EIGER_MX": "application/x-hdf5",
        "AD_EIGER2": "application/x-hdf5",
        "AD_JPEG": "multipart/related;type=image/jpeg",
        "AD_HDF5": "application/x-hdf5",
        "AD_HDF5_GERM": "application/x-hdf5",
        "AD_HDF5_SWMR_STREAM": "application/x-hdf5",
        "AD_HDF5_SWMR_SLICE": "application/x-hdf5",
        "AD_HDF5_SWMR": "application/x-hdf5",
        "AD_HDF5_TS": "application/x-hdf5",    # area_detector_handlers.handlers:AreaDetectorHDF5TimestampHandler
        "AD_HDF5_DET_TS": "application/x-hdf5",    # csx_transforms.AreaDetectorHDF5NDArrayTimeStampHandler
        "AD_TIFF": "multipart/related;type=image/tiff",
        "APB": "application/x-pizzabox-binary",     # columns: timestamp, i0, it, ir, iff, aux1, aux2, aux3, aux4.   iss_patches:APBBinFileHandler
        "APB_TRIGGER": "application/x-pizzabox-binary",  # columns: timestamp, transition,   iss_patches:APBTriggerFileHandler
        "DEX_HDF5": "application/x-hdf5",
        "EIGER2_STREAM": "application/x-hdf5",

        "MERLIN_FLY_STREAM_V2": "application/x-hdf5",
        "MERLIN_HDF5_BULK": "application/x-hdf5",
        "PANDA": "application/x-hdf5",
        "PIL100k_HDF5": "application/x-hdf5",      # iss_patches:ISSPilatusHDF5Handler
        "PILATUS_HDF5": "application/x-hdf5",
        "PIZZABOX_AN_FILE_TXT": "text/csv;header=absent",
        "PIZZABOX_AN_FILE_TXT_PD": "text/csv;header=absent",
        "PIZZABOX_DI_FILE_TXT": "text/csv;header=absent",
        "PIZZABOX_DI_FILE_TXT_PD": "text/csv;header=absent",
        "PIZZABOX_ENC_FILE_TXT": "text/csv;header=absent",
        "PIZZABOX_ENC_FILE_TXT_PD": "text/csv;header=absent",            # iss_patches:PizzaBoxEncHandlerTxtPD
        # "PIZZABOX_FILE": None,   # ISS ???
        "ROI_HDF5_FLY": "application/x-hdf5",
        "ROI_HDF51_FLY": "application/x-hdf5",
        "SIS_HDF51_FLY_STREAM_V1": "application/x-hdf5",
        "TPX_HDF5": "application/x-hdf5",
        
        "NPY_SEQ": "multipart/related;type=application/x-npy",
        "SIS_HDF51": "application/x-hdf5",
        "SPECS_HDF5_SINGLE_DATAFRAME": "application/x-hdf5",    # IOS
        "XIA_XMAP_HDF5": "application/x-hdf5;type=xia-xmap",
        "XSP3": "application/x-hdf5",        # iss_patches:ISSXspress3HDF5Handler, area_detector_handlers.handlers:Xspress3HDF5Handler
        "XSP3_BULK": "application/x-hdf5",
        "XSP3_FLY": "application/x-hdf5",
        "XSP3_STEP": "application/x-hdf5",   # databroker.assets.handlers:Xspress3HDF5Handler, area_detector_handlers.handlers:Xspress3HDF5Handler
        "XSP3X": "application/x-hdf5",
        "ZEBRA_HDF51": "application/x-hdf5",
        "ZEBRA_HDF51_FLY_STREAM_V1": "application/x-hdf5",
    }


# Define document-specific patches to be applied before sending them to TiledWriter

def patch_descriptor(doc):
    if len(doc["data_keys"]) > 750:
        raise NotImplementedError("Descriptors with more than 1000 data keys are not supported yet.")

    # TODO: Only keep the necessary data_keys here
    if "pb1_enc1" in doc["data_keys"]:
        data_key = doc["data_keys"]["pb1_enc1"]
        data_key["dtype_str"] = pb1_dtype.str
        data_key["dtype_descr"] = pb1_dtype.descr
        data_key["shape"] = (LENGTH,)
    if "pb4_di" in doc["data_keys"]:
        data_key = doc["data_keys"]["pb4_di"]
        data_key["dtype_str"] = pb4_dtype.str
        data_key["dtype_descr"] = pb4_dtype.descr
        data_key["shape"] = (LENGTH,)
    if "pb9_enc1" in doc["data_keys"]:
        data_key = doc["data_keys"]["pb9_enc1"]
        data_key["dtype_str"] = pb9_dtype.str
        data_key["dtype_descr"] = pb9_dtype.descr
        data_key["shape"] = (LENGTH,)
    if "apb_stream" in doc["data_keys"]:
        data_key = doc["data_keys"]["apb_stream"]
        data_key["dtype_str"] = apb_dtype.str
        data_key["dtype_descr"] = apb_dtype.descr
        data_key["shape"] = (LENGTH,)
    if "apb_trigger_xs" in doc["data_keys"]:
        data_key = doc["data_keys"]["apb_trigger_xs"]
        data_key["dtype_str"] = apb_trigger_dtype.str
        data_key["dtype_descr"] = apb_trigger_dtype.descr
        data_key["shape"] = (LENGTH,)
    if "apb_trigger_pil100k" in doc["data_keys"]:
        data_key = doc["data_keys"]["apb_trigger_pil100k"]
        data_key["dtype_str"] = apb_trigger_dtype.str
        data_key["dtype_descr"] = apb_trigger_dtype.descr
        data_key["shape"] = (LENGTH,)
    if "apb_trigger_pil100k2" in doc["data_keys"]:
        data_key = doc["data_keys"]["apb_trigger_pil100k2"]
        data_key["dtype_str"] = apb_trigger_dtype.str
        data_key["dtype_descr"] = apb_trigger_dtype.descr
        data_key["shape"] = (LENGTH,)
    if "apb_trigger_ge_detector" in doc["data_keys"]:
        data_key = doc["data_keys"]["apb_trigger_ge_detector"]
        data_key["dtype_str"] = apb_trigger_dtype.str
        data_key["dtype_descr"] = apb_trigger_dtype.descr
        data_key["shape"] = (LENGTH,)
    if "xs_stream" in doc["data_keys"]:
        data_key = doc["data_keys"]["xs_stream"]
        data_key["dtype_str"] = "<f8"
        data_key["shape"] = tuple([1, *data_key.get("shape", (1, 6, 4096))[1:]])
    if "pilatus_image" in doc["data_keys"]:
        data_key = doc["data_keys"]["pilatus_image"]
        data_key["dtype_str"] = "<u2"
    if "pil100k2_image" in doc["data_keys"]:
        data_key = doc["data_keys"]["pil100k2_image"]
        data_key["dtype_str"] = "<u2"
    if "pe1_image" in doc["data_keys"]:
        data_key = doc["data_keys"]["pe1_image"]
        data_key["shape"] = (1, *data_key["shape"][1:])
        data_key["dtype_str"] = "<u2"
    if "pil100k_image" in doc["data_keys"]:    # AD_TIFF spec
        data_key = doc["data_keys"]["pil100k_image"]
        data_key["shape"] = [487, 195]
    for key, val in doc["data_keys"].items():
        if re.match(pba_dkey_pattern, key):
            val["dtype_str"] = pba_dtype.str
            val["dtype_descr"] = pba_dtype.descr
            val["shape"] = (LENGTH,)

    # Ensure dtype_str has the proper numpy format (to pass the EventModel validator)
    for key, val in doc["data_keys"].items():
        if "dtype_str" in val:
            val["dtype_str"] = numpy.dtype(val["dtype_str"]).str
        val["shape"] = tuple(map(lambda x: max(x, 0), val.get("shape", [])))

    for key, val in doc["data_keys"].items():
        if ("external" not in val.keys()) \
            and (val.get("dtype") == "array") \
            and ("filename" in key) \
            and (key not in APB_AVE_FILENAMES):
            raise NotImplementedError(f"Descriptor with external array data key {key} is not supported yet.")

    return doc

def patch_datum(doc):
    # Ensure the resource uid is a string
    if isinstance(doc["resource"], bson.objectid.ObjectId):
        doc["resource"] = str(doc["resource"])

    kwargs = doc.get("datum_kwargs", {})
    spec = kwargs.pop("_resource_spec", None)    # Added by RunNormalizer
    if "data_type" in kwargs:
        if spec == "XIA_XMAP_HDF5":
            # ISSXIAHDF5HandlerLegacy
            if kwargs["data_type"] != "roi":
                raise ValueError("XIA_XMAP_HDF5 only supports data_type='roi'")
            roi_num = kwargs["roi_num"]  # ROI number, 0-indexed
            channel = kwargs["channel"]  # Channel number, 1-indexed
            kwargs["dataset"] = "/entry/data/data"
            kwargs["slice"] = f"(:, {channel-1}, {roi_num})"
            kwargs["squeeze"] = True

        elif kwargs["data_type"] == "image":
            kwargs["dataset"] = "/entry/data/data"

        elif kwargs["data_type"] == "spectrum":
            # databroker.assets.handlers.Xspress3HDF5Handler
            kwargs["dataset"] = "entry/instrument/detector/data"
            channel = kwargs["channel"]     # Assume that 'channel' is always present
            kwargs["slice"] = f"(:,{channel-1},:)"
            kwargs["squeeze"] = True

        elif kwargs["data_type"] == "roi":
            roi_num = kwargs["roi_num"]
            if "channel" in kwargs:
                # ISSXspress3HDF5Handler
                channel = kwargs["channel"]
                kwargs["dataset"] = f"/entry/instrument/detector/NDAttributes/CHAN{channel}ROI{roi_num}"
                kwargs["squeeze"] = True
            else:
                # ISSPilatusHDF5Handler
                kwargs["dataset"] = f"/entry/instrument/NDAttributes/_ROI{roi_num}Total"

    return doc


def patch_resource(doc):

    kwargs = doc.get("resource_kwargs", {})

    # Fix Resource UIDs
    if isinstance(doc["uid"], bson.objectid.ObjectId):
        doc["uid"] = str(doc["uid"])

    # Fix the resource path
    root = doc.get("root", "")
    if not doc["resource_path"].startswith(root):
        doc["resource_path"] = os.path.join(root, doc["resource_path"])
    doc["root"] = ""  # root is redundant if resource_path is absolute
    doc["resource_path"] = doc["resource_path"].replace("/nsls2/xf08id/data", "/nsls2/data/iss/legacy/backup/data")
    doc["resource_path"] = doc["resource_path"].replace("/nsls2/data/iss/legacy/backup/data/electrometer",
                                                        "/nsls2/data/iss/legacy/backup/data/apb")

    if doc.get("spec") in ["PIZZABOX_ENC_FILE_TXT", "PIZZABOX_ENC_FILE_TXT_PD",
                           "PIZZABOX_AN_FILE_TXT", "PIZZABOX_AN_FILE_TXT_PD",
                           "PIZZABOX_DI_FILE_TXT", "PIZZABOX_DI_FILE_TXT_PD",
                           "APB", "APB_TRIGGER"]:
        kwargs.update({"sep": " "})    # Data are in space-separated csv format
    elif doc.get("spec") in ["XSP3", "XSP3X"]:
        kwargs.update({"dataset": 'entry/instrument/detector/data', "chunk_shape": (1, ), "join_method": "concat"})
    elif doc.get("spec") in ["AD_HDF5", "AD_HDF5_SWMR_STREAM", "AD_HDF5_SWMR_SLICE", "AD_HDF5_SWMR", "PIL100k_HDF5", "PILATUS_HDF5"]:
        kwargs.update({"dataset": 'entry/instrument/detector/data'})
    elif doc.get("spec") in ["AD_TIFF"]:
        kwargs["template"] = "/" + kwargs["template"].lstrip("/")    # Ensure leading slash
        kwargs["join_method"] = "stack"

    return doc

# TODO: The RunNormalizer is likely unnecessary if all APB streams are correctly declared in Resources

class RunNormalizerISS(RunNormalizer):

    fname_keys = APB_AVE_FILENAMES  # data keys that store filepaths as encrypted arrays and need to be fixed

    def __init__(self, *args, **kwargs):
        self.apb_stream_fpath_bin = None
        self._specs_by_resource_uid = {}
        super().__init__(*args, **kwargs)

    def produce_apb_stream(self, fpath_bin, timestamp):
        desc_doc = {
            "run_start": self.run_start_uid,
            "name": "apb_stream",
            "time": timestamp,
            "uid": str(uuid.uuid4()),
            "data_keys": {
                "apb_stream": {
                    "source": "APB",
                    "dtype": "array",
                    "external": "FILESTORE:",
                    "dtype_str": apb_dtype.str,
                    "dtype_descr": apb_dtype.descr,
                    "shape": (LENGTH,)
                }
            },
            "configuration": {
                "apb_stream-pb1_enc1-flyer": {
                    "data": {},
                    "timestamps": {},
                    "data_keys": {}
                }
            },
            "hints": {},
            "object_keys": {
                "apb_stream-pb1_enc1-flyer": [
                    "apb_stream"
                ]
            }
        }
        resource_doc = {
            "spec": "APB",
            "resource_path": fpath_bin,
            "root": "",
            "resource_kwargs": {},
            "path_semantics": "posix",
            "uid": str(uuid.uuid4()),
            "run_start": self.run_start_uid
        }
        datum_doc = {
                "resource": resource_doc["uid"],
                "datum_id": f"{resource_doc['uid']}/0",
                "datum_kwargs": {}
            }
        event_doc = {
                "descriptor": desc_doc["uid"],
                "uid": str(uuid.uuid4()),
                "data": {
                    "apb_stream": f"{resource_doc['uid']}/0"
                },
                "timestamps": {
                    "apb_stream": timestamp
                },
                "time": timestamp,
                "seq_num": 1
            }

        self.descriptor(desc_doc)
        self.resource(resource_doc)
        self.datum(datum_doc)
        self.event(event_doc)

        self.notes.append(f"Data stream 'apb_stream' has been added to register data from an APB file referenced in 'primary'.")

    def start(self, doc):
        self.run_start_uid = doc["uid"]
        return super().start(doc)
    
    def descriptor(self, doc):
        # Fix the filename paths in apb_ave_filename_* data keys
        if set(doc.get("data_keys", {}).keys()).intersection(self.fname_keys):
            doc = copy.deepcopy(doc)
            for key in self.fname_keys:
                if val := doc.get("data_keys", {}).get(key):
                    if val["dtype"] == "array":
                        val["dtype"] = "string"
                        val["shape"] = []
                        val["units"] = None
                        val["lower_ctrl_limit"] = None
                        val["upper_ctrl_limit"] = None

        return super().descriptor(doc)

    def resource(self, doc):
        # Keep track of spec by resource uid
        self._specs_by_resource_uid[doc["uid"]] = doc.get("spec")

        return super().resource(doc)
    
    def event(self, doc):
        if set(doc["data"].keys()).intersection(self.fname_keys):
            doc = copy.deepcopy(doc)
            for key in self.fname_keys:
                if val := doc["data"].get(key):
                    doc["data"][key] = "".join((chr(int(i)) for i in val if i > 0))

            # Keep track of apb_stream file path for later use
            self.apb_stream_fpath_bin = self.apb_stream_fpath_bin \
                        or doc["data"].get("apb_ave_filename_bin")
                        # or doc["data"].get("apb_ave_c_filename_bin") \
                        # or doc["data"].get("apb_filename_bin") \
                        # or doc["data"].get("apb_c_filename_bin")
            if self.apb_stream_fpath_bin and self.apb_stream_fpath_bin.startswith("/home"):
                self.apb_stream_fpath_bin = None

        return super().event(doc)
    
    def datum(self, doc):
        # Make note of the resource spec in datum_kwargs
        doc = copy.deepcopy(doc)

        if spec := self._specs_by_resource_uid.get(doc["resource"]):
            doc["datum_kwargs"]["_resource_spec"] = spec

        return super().datum(doc)
    
    def stop(self, doc):
        if self.apb_stream_fpath_bin and Path(self.apb_stream_fpath_bin).is_file():
            self.produce_apb_stream(self.apb_stream_fpath_bin, doc["time"] - 0.0001)

        super().stop(doc)


# Define a custom consolidator for PizzaBox binary files
# NOTE: Probably Unnecessary
from bluesky_tiled_plugins.writing.consolidators import CONSOLIDATOR_REGISTRY, ConsolidatorBase

class PizzaBoxConsolidator(ConsolidatorBase):
    supported_mimetypes: set[str] = {"application/x-pizzabox-binary"}

CONSOLIDATOR_REGISTRY["application/x-pizzabox-binary"] = PizzaBoxConsolidator


# Initialize the Tiled client and the TiledWriter
api_key = os.environ.get("TILED_BLUESKY_WRITING_API_KEY_ISS")
tiled_writing_client_sql = from_uri("https://tiled.nsls2.bnl.gov", api_key=api_key)['iss/migration']
tw = TiledWriter(client = tiled_writing_client_sql,
                 backup_directory="/tmp/tiled_backup",
                 patches = {"descriptor": patch_descriptor,
                            "resource": patch_resource,
                            "datum": patch_datum},
                 spec_to_mimetype = MIMETYPE_LOOKUP,
                 batch_size=10000   # NOTE: Set to 1 to disable batching
                 )

# Thread-safe wrapper for TiledWriter
tw = BufferingWrapper(tw)

# Subscribe the TiledWriter
RE.md["tiled_access_tags"] = (RE.md["data_session"],)
RE.subscribe(tw)
