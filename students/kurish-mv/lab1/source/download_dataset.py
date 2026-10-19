import kagglehub
from kagglehub import KaggleDatasetAdapter

file_path = ""

df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "kumargh/pimaindiansdiabetescsv",
  file_path,
)
