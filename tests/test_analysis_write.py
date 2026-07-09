import matplotlib
matplotlib.use("Agg")

import uuid


# TEST root
ISSANALYSISROOT = "/tmp"
ISSANALYSISSPEC = 'ISS-hdf5-df'
# dict of key, writer, and callable to make resource args
external_writers = {'interp_df' : {'writer': HDF5DFWriter, 
                                    'resource': {'func': make_resource_hdf5dfwriter,
                                    'kwargs' : dict(root=ISSANALYSISROOT)},
                                    'root' : ISSANALYSISROOT,
                                    'spec' : ISSANALYSISSPEC,
                                    },
                    'bin_df' : {'writer': HDF5DFWriter, 
                                    'resource': {'func': make_resource_hdf5dfwriter,
                                    'kwargs' : dict(root=ISSANALYSISROOT)},
                                    'root' : ISSANALYSISROOT,
                                    'spec' : ISSANALYSISSPEC,
                                    },
                    }


md = dict(sample_name="foo", user="Julien", cycle="2018-1", proposal="00000")
# add more data
md['parent_uid'] = ""
md['plan_name'] = 'analysis'
md['name'] = 'analysis'

# test data
import pandas as pd
import h5py

df_dict = [
            dict(a=1, b=2, c=3),
            dict(a=2, b=2, c=3),
            dict(a=1, b=1, c=3),
            dict(a=1, b=2, c=4),
            ]

df = pd.DataFrame(df_dict)

df.to_hdf("foo.hd5", "df1")

data_dicts = [
        dict(primary=dict(interp_df=df))
        ]

doc_gen = ingest(data_dicts, md=md, external_writers=external_writers)


dbname = 'iss-analysis'
todb = ToDataBroker(dbname)
for nds in doc_gen:
    todb(*nds)

db_analysis = Broker.named("iss-analysis")
# Not Critical but since this isn't a workflow it's permitted
db_analysis.v2.context.http_client.headers['tiled-qos'] = 'acquisition'
db_analysis.reg.register_handler(ISSANALYSISSPEC, HDF5DFHandler)
