import kagglehub

file_path = "students\\kurish-mv\\lab1\\source\\dataset"

path = kagglehub.dataset_download("kumargh/pimaindiansdiabetescsv", output_dir=file_path)

print("Path to dataset files:", path)