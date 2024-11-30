import requests
import xml.etree.ElementTree as ET
import tarfile
import os

def extract_and_delete_tar_gz(file_path, extract_to):
    """
    解压 tar.gz 文件到指定文件夹，并删除原文件

    :param file_path: tar.gz 文件的路径
    :param extract_to: 解压目标文件夹
    """
    # 检查文件是否为 tar.gz 格式
    if not tarfile.is_tarfile(file_path):
        print(f"{file_path} 不是一个有效的 tar.gz 文件")
        return
    
    # 创建目标文件夹（如果不存在）
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    
    try:
        # 解压文件
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=extract_to)
        print(f"文件已解压到 {extract_to}")
        
        # # 删除原文件
        # os.remove(file_path)
        # print(f"已删除原文件: {file_path}")
    except Exception as e:
        print(f"处理文件时出错: {e}")

def search_arxiv_paper(title):
    # 构造查询URL
    query_url = f"http://export.arxiv.org/api/query?search_query=ti:{title}&start=0&max_results=1"
    response = requests.get(query_url)
    if response.status_code != 200:
        raise Exception(f"Failed to search arXiv: {response.status_code}")
    
    # 解析返回的XML
    root = ET.fromstring(response.text)
    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
        arxiv_id = entry.find('{http://arxiv.org/schemas/atom}id').text.split('/')[-1]
        return arxiv_id
    raise Exception("No paper found for the given title.")

def download_arxiv_latex(arxiv_id, output_file):
    # 下载LaTeX源代码
    download_url = f"https://arxiv.org/e-print/{arxiv_id}"
    response = requests.get(download_url)
    if response.status_code != 200:
        raise Exception(f"Failed to download LaTeX source: {response.status_code}")
    
    # 保存为文件
    with open(output_file, 'wb') as f:
        f.write(response.content)
    print(f"Downloaded LaTeX source to {output_file}")

if __name__ == '__main__':
    # 示例使用
    #RAG based Question-Answering for Contextual Response Prediction System
    arxiv_id = [
        '2409.02574',
        '2409.03171',
        '2409.02838',
        '2409.02284',
        '2409.02310',
        '2409.02481',
        '2409.02566',
        '2409.02335',
        '2409.03223',
        '2409.02611',
        '2409.02370',
        '2409.02699',
        '2409.03530',
        '2409.02078',
        '2409.03458',
        '2409.03062',
        '2409.02465',
        '2409.03327',
        '2409.03346',
        '2409.03550',
        '2409.03183',
        '2409.03701',
        '2409.03236',
        '2409.03456',
        '2409.02825',
        '2409.02438',
        '2409.03516',
        '2409.02261',
        '2409.03708',

    ]
    paper_title = '"SegTalker: Segmentation-based Talking Face Generation with \\ Mask-guided Local Editing"'  # 替换为论文标题
    
    for id in arxiv_id:
        try:
            # print(f"Found arXiv ID: {id}")
            # download_arxiv_latex(id, f"./data/paper_hf/CS/latex/{id}.tar.gz")
            extract_and_delete_tar_gz(f"./data/paper_hf/CS/latex/{id}.tar.gz", f"./data/paper_hf/CS/latex/{id}/")
            input()
        except Exception as e:
            print(e)
