
TAE-HAIC Objective Response Prediction Web App
中英文双语版 / Chinese-English bilingual version

文件说明：
1. app.py
   双语网页主程序。

2. catboost_model.pkl
   已训练好的CatBoost模型。

3. score_thresholds.json
   基于训练集pred_score生成的Q1-Q4分组截点。

4. requirements.txt
   Python依赖包。

5. install_requirements.bat
   安装依赖包。

6. run_app.bat
   启动网页。

首次运行：
1. 双击 install_requirements.bat 安装依赖。
2. 双击 run_app.bat 启动网页。

或者在命令行中运行：
cd /d D:/qq/预后分析/TAE_HAIC_web_app
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501

网页功能：
- 支持中文 / English语言切换
- 输入8个治疗前变量
- 输出客观缓解预测概率
- 输出Q1-Q4预测评分分组
- 输出简要解释

注意：
该网页仅用于科研展示和辅助评估，不能替代临床决策。
