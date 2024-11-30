import re
import json

def json_save(data, path):
    f = open(path, 'w', encoding = 'utf-8')
    json.dump(data, f)
    f.close()


def read_tex_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except Exception as e:
        print(f"Error reading the file: {e}")
        return None

def read_bbl_file(file_path):
    """
    读取 .bbl 文件并返回内容
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
        return None
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return None

def extract_citations_from_bbl(content):
    """
    从 .bbl 文件内容中提取参考文献条目
    """
    citations = []
    if content:
        lines = content.splitlines()
        in_citation = False
        citation = []
        
        for line in lines:
            if "\\bibitem" in line:  # 检测参考文献条目开始
                if citation:  # 保存之前的参考文献条目
                    citations.append("\n".join(citation))
                    citation = []
                in_citation = True
            if in_citation:
                citation.append(line)
        
        if citation:  # 保存最后一个参考文献条目
            citations.append("\n".join(citation))
    return citations

if __name__ == '__main__':
    import os

    latex_files = os.listdir('./data/paper_hf/CS/latex/')

    for lf in latex_files:
        if '.tar.gz' not in lf:
            project_files = os.listdir(f'./data/paper_hf/CS/latex/{lf}')
            for pf in project_files:
                if '.tex' in pf:
                    tex_content = read_tex_file(f'./data/paper_hf/CS/latex/{lf}/{pf}')
                
                if '.bbl' in pf:
                    file_path = f'./data/paper_hf/CS/latex/{lf}/{pf}'
                    bbl_content = read_bbl_file(file_path)


        pattern = re.compile(r"\\begin{table.*?\\end{table", re.DOTALL)

        # Find all matches
        experiment_results = []
        tables = pattern.findall(tex_content)

        for i, table in enumerate(tables, 1):
            experiment_results.append(table)

        sections = tex_content.split('\section{')

        for s in sections:
            if 'related work' in s.lower():
                cites_index1 = re.findall(r'~\\citep\{(.*?)\}', s)
                cites_index2 = re.findall(r'~\\citet\{(.*?)\}', s)
                cites_index3 = re.findall(r'~\\cite\{(.*?)\}', s)

                cites_index = []
                for item in cites_index1:
                    if ',' in item:
                        cites_index.extend(item.split(','))
                    else:
                        cites_index.append(item)
                
                for item in cites_index2:
                    if ',' in item:
                        cites_index.extend(item.split(','))
                    else:
                        cites_index.append(item)
                
                for item in cites_index3:
                    if ',' in item:
                        cites_index.extend(item.split(','))
                    else:
                        cites_index.append(item)


        Related_Work_Ref = []
        if bbl_content:
            citations = extract_citations_from_bbl(bbl_content)

            for ci in cites_index:
                for c in citations:
                    if ci in c:
                        Related_Work_Ref.append(c)
                        break
        
        json_save(Related_Work_Ref, f'./data/paper_hf/CS/latex/{lf}/Related_Work_Ref.json')
        json_save(experiment_results, f'./data/paper_hf/CS/latex/{lf}/experiment_results.json')