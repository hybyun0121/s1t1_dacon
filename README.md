# Samsung AI Challenge for Scientific Discovery

Dacon 대회 참여 코드

### Graph data로 변환

분자 데이터 정보가 smiles type으로 주어져있다.
rdkit 라이브러리를 이용하여 분자 특성을 추출하여
pytorch geometric 의 graph type으로 변환해주었다.

### Model

사용한 model 은 CGConv이다.
[CGCN-리뷰](https://github.com/hybyun0121/gnn-pr/blob/main/paper-review/CGCNN-리뷰.pdf)
[CGConv- PyG](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/cg_conv.html#CGConv)

### run code

~~~python
python main.py
~~~

### 학습결과

<img src="https://user-images.githubusercontent.com/63500940/136166740-ac859ed4-fb81-4406-b252-47ae5b1c0e55.png" alt="plot" style="zoom:80%;" />

