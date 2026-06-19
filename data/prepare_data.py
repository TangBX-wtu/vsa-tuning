# 确保json文件已经符合加载格式
from preprocess import VSAPreprocessor

preprocessor = VSAPreprocessor(
    model_name="./CodeBert-pretrained",
    max_seq_len=512,
    joern_home="D:\\tools\\joern-cli"
)

# preprocessor.process_file("./original_dataset_preprocess/SARD/dataset/augmented/augmented_no_kl_sard_test.json", "./processed/SARD/SARD_AUG_NO_KL_test_v6.pt")
# preprocessor.process_file("./original_dataset_preprocess/SARD/dataset/augmented/augmented_no_kl_sard_train.json", "./processed/SARD/SARD_AUG_NO_KL_train_v6.pt")
# preprocessor.process_file("./original_dataset_preprocess/SARD/dataset/augmented/augmented_no_kl_sard_val.json", "./processed/SARD/SARD_AUG_NO_KL_val_v6.pt")

# preprocessor.process_file("./original_dataset_preprocess/CLEANVUL/CLEANVUL_NO_KEYLINE_FIXED.json", "./processed/CLEANVUL/CLEANVUL_NO_KEYLINE_FIXED_v6.pt")
# preprocessor.process_file("./original_dataset_preprocess/PRIMEVUL/PRIMEVUL_NO_KEYLINE_FIXED.json", "./processed/PRIMEVUL/PRIMEVUL_NO_KEYLINE_FIXED_v6.pt")
# preprocessor.process_file("./original_dataset_preprocess/TITANVUL/TITANVUL_NO_KEYLINE_FIXED.json", "./processed/TITANVUL/TITANVUL_NO_KEYLINE_FIXED_v6.pt")
# preprocessor.process_file("./original_dataset_preprocess/DIVERSEVUL/DIVERSEVUL_NO_KEYLINE_FIXED.json", "./processed/DIVERSEVUL/DIVERSEVUL_NO_KEYLINE_FIXED_v6.pt")

# preprocessor.process_file("./SRTS/UAF/PRIMEVUL/UAF_ORIGIN.json", "./processed/SRTS/UAF/PRIMEVUL/UAF_ORIGIN.pt")
# preprocessor.process_file("./SRTS/UAF/PRIMEVUL/UAF_REVERSAL.json", "./processed/SRTS/UAF/PRIMEVUL/UAF_REVERSAL.pt")

# preprocessor.process_file("./SRTS/UAF/TITANVUL/UAF_ORIGIN.json", "./processed/SRTS/UAF/TITANVUL/UAF_ORIGIN.pt")
# preprocessor.process_file("./SRTS/UAF/TITANVUL/UAF_REVERSAL.json", "./processed/SRTS/UAF/TITANVUL/UAF_REVERSAL.pt")

preprocessor.process_file("./SRTS/UAF/SARD/UAF_ORIGIN.json", "./processed/SRTS/UAF/SARD/UAF_ORIGIN.pt")
preprocessor.process_file("./SRTS/UAF/SARD/UAF_REVERSAL.json", "./processed/SRTS/UAF/SARD/UAF_REVERSAL.pt")