import nlpaug.augmenter.word as naw
# aug = naw.ContextualWordEmbsAug(
#     model_path=model_name, action="insert", device=device)

# Augment French by BERT
# bert-base-multilingual-uncased  yechen/bert-large-chinese
# aug = naw.ContextualWordEmbsAug(model_path='bert-base-multilingual-uncased', action="insert", aug_p=0.1)
aug = naw.ContextualWordEmbsAug(model_path='./PLM/bert-base-chinese', action="insert", device='cuda', aug_p=0.1, batch_size=128)
text = "你的声音好好听啊   六六六，兄弟你可真会说话！"
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)