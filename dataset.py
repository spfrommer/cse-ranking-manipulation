import argparse

import re
import typing as t
import pandas as pd
import os
import csv

from bs4 import BeautifulSoup
import tqdm

from _types import (
    ChatFunction,
    Message,
    Role
)

from models import Models, load_model
from prompts import get_prompt_for_dataset_rewrite, get_prompt_for_body_extraction
from _types import Product
from helpers import file_utils

DATASET_DIR = './dataset'

def user_query(category: str) -> str:
    return f"I'm looking to buy a {category}. Can you give me some recommendations?"

def get_categories(dataset_dir=DATASET_DIR) -> t.List[str]:
    return [
        d for d in os.listdir(dataset_dir) if
        os.path.isdir(os.path.join(dataset_dir, d))
    ]


def get_products(
        category: str,
        dataset_dir=DATASET_DIR,
        returned_doc='content_truncate'
    ) -> t.List[t.Tuple[Product, str, int]]:
    # Each tuple has the product, document, and index in original csv.
    category_dir = os.path.join(dataset_dir, category)
    doc_dir = os.path.join(category_dir, returned_doc)
    df = pd.read_csv(os.path.join(category_dir, 'products.csv'))
    
    products = []
    
    for file in os.listdir(doc_dir):
        full_path = os.path.join(doc_dir, file)
        with open(full_path, 'r') as f:
            doc = f.read()
        
        csv_index = int(file.split('.')[0])
        
        row = df.iloc[csv_index]
        product = Product(
            category=row['Product'], brand=row['Brand'], model=row['Model']
        )
        
        products.append((product, doc, csv_index))

    return products

def get_products_with_rewritten_docs(
        category: str,
        dataset_dir=DATASET_DIR,
    ) -> t.Tuple[t.List[Product], t.List[t.List[str]]]:
    # First is a list of products
    # Second is a list of lists of documents
    # Outside list is the "target" product that should be injected into the doc
    # Inside list is the documents

    category_dir = os.path.join(dataset_dir, category)
    doc_dir = os.path.join(category_dir, 'content_rewrite')
    df = pd.read_csv(os.path.join(category_dir, 'products.csv'))
    
    products, docs = [], []
    
    for i in range(len(df)):
        row = df.iloc[i]
        products.append(Product(
            category=row['Product'], brand=row['Brand'], model=row['Model']
        ))
        
        rewritten_docs = []
        for j in range(len(df)):
            full_path = os.path.join(doc_dir, str(i), f'{j}.txt')
            with open(full_path, 'r') as f:
                doc = f.read()
            rewritten_docs.append(doc)
            
        docs.append(rewritten_docs)

    return products, docs



def reextract_content(args: argparse.Namespace):
    for category in get_categories():
        category_dir = os.path.join(DATASET_DIR, category)
        pages_dir = os.path.join(category_dir, 'pages')
        content_dir = os.path.join(category_dir, 'content')
        file_utils.create_empty_directory(content_dir)
        
        for file in os.listdir(pages_dir):
            full_path = os.path.join(pages_dir, file)
            with open(full_path, 'r') as f:
                soup = BeautifulSoup(f, 'html.parser')
            
            content = soup.get_text(separator='\n')
            
            i = int(file.split('.')[0])
            
            save_path = os.path.join(content_dir, f'{i}.txt')
            
            with open(save_path, 'w') as f:
                f.write(content)

def truncate_products(args: argparse.Namespace):
    truncator = load_model(
        args.truncator_model,
        args.truncator_temp,
        args.truncator_top_p,
        args.truncator_max_tokens
    )
    for category in get_categories():
        print(category)
        category_dir = os.path.join(DATASET_DIR, category)
        content_extract_dir = os.path.join(category_dir, 'content_extract')
        content_trunc_dir = os.path.join(category_dir, 'content_truncate')
        
        file_utils.create_empty_directory(content_extract_dir)
        file_utils.create_empty_directory(content_trunc_dir)
        
        for product, doc, index in get_products(category, returned_doc='content'):
            doc = re.sub(r'\n{2,}', '\n\n', doc)
            
            prompt = get_prompt_for_body_extraction(
                doc=doc, brand=product.brand, model=product.model
            )
            
            extracted_doc = truncator(
                [Message(role=Role.user, content=prompt)]
            ).content
            
            with open(os.path.join(content_extract_dir, f'{index}.txt'), 'w') as f:
                f.write(extracted_doc)
            
            truncated_doc = extracted_doc[:args.truncate_length]
            
            with open(os.path.join(content_trunc_dir, f'{index}.txt'), 'w') as f:
                f.write(truncated_doc)
            
def rewrite_product_documents(args: argparse.Namespace):
    rewriter = load_model(
        args.rewriter_model,
        args.rewriter_temp,
        args.rewriter_top_p,
        args.rewriter_max_tokens
    )

    for category in tqdm.tqdm(get_categories()):
        products = get_products(category)
        
        rewrite_dir = os.path.join(DATASET_DIR, category, 'content_rewrite')
        
        # Subdirs are the product indices
        
        for product_new, _, index_new in products:
            subdir = os.path.join(rewrite_dir, str(index_new))
            file_utils.ensure_created_directory(subdir)

            for product_old, doc_old, index_old in products:
                if index_new == index_old:
                    rewritten_doc = doc_old
                else:
                    prompt = get_prompt_for_dataset_rewrite(
                        doc=doc_old,
                        brand_old=product_old.brand,
                        model_old=product_old.model,
                        brand_new=product_new.brand,
                        model_new=product_new.model
                    )
                    
                    rewritten_doc = rewriter(
                        [Message(role=Role.user, content=prompt)]
                    ).content

                with open(os.path.join(subdir, f'{index_old}.txt'), 'w') as f:
                    f.write(rewritten_doc)

    
def add_reextract_content_arguments(parser):
    parser.add_argument('--reextract-content', action='store_true')
    
def add_truncate_arguments(parser):
    parser.add_argument('--truncate', action='store_true')
    parser.add_argument('--truncate-length', type=int, default=1000)
    
    parser.add_argument(
        "--truncator-model", type=str, default="gpt-3.5", choices=Models.keys(), help="Truncator model"
    )
    parser.add_argument("--truncator-temp", type=float, default=0.1, help="Truncator temperature")
    parser.add_argument("--truncator-top-p", type=float, default=1.0, help="Truncator top-p")
    parser.add_argument("--truncator-max-tokens", type=int, default=1024, help="Truncator max tokens")

def add_rewriter_arguments(parser):
    parser.add_argument('--rewrite', action='store_true')
    parser.add_argument(
        "--rewriter-model", type=str, default="llama3-70b", choices=Models.keys(), help="Rewriter model"
    )
    parser.add_argument("--rewriter-temp", type=float, default=0.2, help="Rewriter temperature")
    parser.add_argument("--rewriter-top-p", type=float, default=1.0, help="Rewriter top-p")
    parser.add_argument("--rewriter-max-tokens", type=int, default=1024, help="Rewriter max tokens")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
        
    add_reextract_content_arguments(parser)
    add_truncate_arguments(parser)
    add_rewriter_arguments(parser)
    

    args = parser.parse_args()
    
    assert [args.reextract_content, args.truncate, args.rewrite].count(True) == 1
    
    if args.reextract_content:
        reextract_content(args)
    elif args.truncate:
        truncate_products(args)
    elif args.rewrite:
        rewrite_product_documents(args)
