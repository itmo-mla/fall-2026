def generate_report(replace_dict):
    with open("./docs/template_README.md", "r") as f:
        template_text = f.read()

    for k, v in replace_dict.items():
        v = str(v)
        template_text = template_text.replace(f"$${k}$$", v)
    if "$$" in template_text:
        print("Not all data passed to report generator.")
    with open("gen_README.md", "w") as f:
        f.write(template_text)


if __name__ == "__main__":
    generate_report(
        {
            "dataset_desc": "Для выполнения лабораторной работы был выбран датасет  [донорство крови](https://www.kaggle.com/datasets/vstacknocopyright/blood-transfusion-service-center-data)",
            "h_best_my": 4,
            "kernel_best_my": "rect",
            "llo_best_my": 0.2273,
            "h_best_sk": 6,
            "kernel_best_sk": "uniform",
            "llo_best_sk": 0.2246,
            "full_size": 748,
            "full_loo": 0.2273,
            "ref_size": 115,
            "ref_size_percentage": 15.4,
            "ref_loo": 0.1725,
        }
    )
