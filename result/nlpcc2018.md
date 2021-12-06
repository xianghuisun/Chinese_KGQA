version1版本指的是没经过任何代码修改的版本，参数配置上也没有进行更改。

version2版本指的是将model.py中的loss改为MSE损失，即loss=torch.nn.MSELoss()(last_e,answers)



|          | nlpcc2018 | WebQSP |
| -------- | --------- | ------ |
| version1 | 0.738     |        |
| version2 | 0.714     |        |
|          |           |        |

