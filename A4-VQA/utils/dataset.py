# utils/dataset.py
import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer, BertModel

class CLEVRA4Dataset(Dataset):
    def __init__(self, image_dir, question_json, tokenizer_path=None, max_len=30, mode='train'):
        """
        image_dir: path to images
        question_json: path to questions.json
        tokenizer_path: if using local BERT, otherwise None
        max_len: maximum length for question tokens
        mode: 'train' or 'test'
        """
        self.image_dir = image_dir
        self.questions = self.load_questions(question_json)
        self.max_len = max_len# Ensure max_len is at least the computed max length
        self.mode = mode
        
        # Load tokenizer
        if tokenizer_path:
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
            model = BertModel.from_pretrained(tokenizer_path)
            self.tokenizer.save_pretrained('local_bert')
            model.save_pretrained('local_bert', safe_serialization=False)   
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertModel.from_pretrained('bert-base-uncased')
            self.tokenizer.save_pretrained('local_bert')
            model.save_pretrained('local_bert', safe_serialization=False)    
            
        # Image Transformations (same as ResNet expects)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Create answer to index mapping
        self.answer2idx = self.create_answer_vocab()


    def load_questions(self, question_json):
        with open(question_json, 'r') as f:
            data = json.load(f)
        return data['questions']
    
    def create_answer_vocab(self):
        # Build a set of all possible answers
        all_answers = set()
        for q in self.questions:
            all_answers.add(q['answer'])
        answer2idx = {ans: idx for idx, ans in enumerate(sorted(all_answers))}
        return answer2idx
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        while True:  # Keep trying until a valid image is loaded
            question_data = self.questions[idx]
            image_filename = question_data['image_filename']
            question_text = question_data['question']
            answer = question_data['answer']

            # Load and transform image
            img_path = os.path.join(self.image_dir, image_filename)
            try:
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)
            except (FileNotFoundError, OSError):
                # Randomly sample a new index if file not found or image is corrupted
                idx = torch.randint(0, len(self.questions), (1,)).item()
                continue  # retry loading
            
            # Tokenize question
            tokens = self.tokenizer(
                question_text, 
                padding='max_length', 
                truncation=True, 
                max_length=self.max_len,
                return_tensors='pt'
            )
            input_ids = tokens['input_ids'].squeeze(0)   # [max_len]
            attention_mask = tokens['attention_mask'].squeeze(0) # [max_len]

            # Label encoding
            label = self.answer2idx[answer]

            return {
                'image': image,                    # [3, 224, 224]
                'input_ids': input_ids,             # [max_len]
                'attention_mask': attention_mask,   # [max_len]
                'label': torch.tensor(label)        # scalar
            }

    def get_answer_vocab(self):
        return self.answer2idx