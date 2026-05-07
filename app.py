
# -*- coding: utf-8 -*-

import json
import pickle
import pandas as pd
import streamlit as st


# ============================================================
# 1. 页面设置
# ============================================================

st.set_page_config(
    page_title="TAE-HAIC Response Prediction Model",
    page_icon="🧬",
    layout="centered"
)

MODEL_PATH = "catboost_model.pkl"
THRESHOLD_PATH = "score_thresholds.json"


# ============================================================
# 2. 加载模型和分组截点
# ============================================================

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model


@st.cache_data
def load_thresholds():
    with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
        thresholds = json.load(f)
    return thresholds


model = load_model()
thresholds = load_thresholds()
feature_vars = list(model.feature_names_)


# ============================================================
# 3. 中英文语言选择
# ============================================================

LANGUAGE = st.sidebar.selectbox(
    "Language / 语言",
    ["中文", "English"]
)

IS_CN = LANGUAGE == "中文"


def text(cn, en):
    return cn if IS_CN else en


# ============================================================
# 4. 分组和预测函数
# ============================================================

def assign_score_group(score, thresholds):
    q25 = thresholds["q25"]
    q50 = thresholds["q50"]
    q75 = thresholds["q75"]

    if score <= q25:
        return "Q1"
    elif score <= q50:
        return "Q2"
    elif score <= q75:
        return "Q3"
    else:
        return "Q4"


def get_group_interpretation(score_group):
    if score_group == "Q1":
        return text(
            "低预测评分组：该患者接受TAE-HAIC后获得客观缓解的可能性较低。",
            "Low prediction-score group: this patient has a relatively low probability of achieving objective response after TAE-HAIC."
        )

    if score_group == "Q2":
        return text(
            "中低预测评分组：该患者接受TAE-HAIC后获得客观缓解的可能性偏低至中等。",
            "Low-intermediate prediction-score group: this patient has a low-to-intermediate probability of achieving objective response after TAE-HAIC."
        )

    if score_group == "Q3":
        return text(
            "中高预测评分组：该患者接受TAE-HAIC后获得客观缓解的可能性中等偏高。",
            "Intermediate-high prediction-score group: this patient has an intermediate-to-high probability of achieving objective response after TAE-HAIC."
        )

    return text(
        "高预测评分组：该患者接受TAE-HAIC后获得客观缓解的可能性较高。",
        "High prediction-score group: this patient has a relatively high probability of achieving objective response after TAE-HAIC."
    )


def get_probability_of_class_1(model, X):
    proba = model.predict_proba(X)

    if hasattr(model, "classes_"):
        classes = list(model.classes_)

        class_index = None
        for target_label in [1, "1", 1.0]:
            if target_label in classes:
                class_index = classes.index(target_label)
                break

        if class_index is None:
            class_index = 1
    else:
        class_index = 1

    return float(proba[:, class_index][0])


# ============================================================
# 5. 页面标题
# ============================================================

st.title(
    text(
        "TAE-HAIC客观缓解预测模型",
        "TAE-HAIC Objective Response Prediction Model"
    )
)

st.markdown(
    text(
        """
本网页用于预测原发性肝癌患者接受 **TAE-HAIC** 治疗后的客观缓解概率。

**结局定义：**

- CR / PR = 1  
- SD / PD = 0  

模型输出值为患者达到客观缓解，即 **CR/PR** 的预测概率。
""",
        """
This web calculator estimates the probability of achieving objective response after **TAE-HAIC** treatment in patients with hepatocellular carcinoma.

**Outcome definition:**

- CR / PR = 1  
- SD / PD = 0  

The model output represents the predicted probability of objective response, namely **CR/PR**.
"""
    )
)


with st.expander(
    text(
        "变量定义说明：Vp分级与Up-to-seven标准",
        "Variable definitions: Vp classification and up-to-seven criteria"
    ),
    expanded=True
):
    st.markdown(
        text(
            """
**Vp分级定义：**

- **Vp0**：无门静脉癌栓；
- **Vp1**：癌栓位于门静脉二级分支以远；
- **Vp2**：癌栓累及门静脉二级分支；
- **Vp3**：癌栓累及门静脉一级分支；
- **Vp4**：癌栓累及门静脉主干或对侧门静脉分支。

**Up-to-seven标准：**

Up-to-seven标准根据 **最大肿瘤直径（cm） + 肿瘤数目** 计算。  
若二者之和 **≤ 7**，通常认为符合Up-to-seven标准；若 **> 7**，则不符合该标准。

**编码提醒：**

本网页输入值必须与原始建模数据中的编码保持一致。本网页默认：  
- `up_to_seven = 1` 表示符合Up-to-seven标准；  
- `up_to_seven = 0` 表示不符合Up-to-seven标准。
""",
            """
**Vp classification:**

- **Vp0**: no portal vein tumor thrombus;
- **Vp1**: tumor thrombus distal to the second-order branches of the portal vein;
- **Vp2**: tumor thrombus involving the second-order branches;
- **Vp3**: tumor thrombus involving the first-order branches;
- **Vp4**: tumor thrombus involving the main portal vein trunk or the contralateral portal vein branch.

**Up-to-seven criteria:**

The up-to-seven criteria are calculated as:  
**largest tumor diameter in centimeters + number of tumors**.  
A sum of **≤ 7** is generally considered within the up-to-seven criteria, whereas a sum of **> 7** is considered beyond the criteria.

**Coding note:**

The input values must be consistent with the coding used in the original modeling dataset. This web calculator assumes that:  
- `up_to_seven = 1` indicates within the up-to-seven criteria;  
- `up_to_seven = 0` indicates beyond the up-to-seven criteria.
"""
        )
    )

st.divider()


# ============================================================
# 6. 输入变量
# ============================================================

st.subheader(
    text(
        "请输入患者治疗前临床变量",
        "Input pretreatment clinical variables"
    )
)

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:

        Tumor_number = st.number_input(
            text("肿瘤数目", "Tumor number"),
            min_value=1,
            max_value=50,
            value=3,
            step=1,
            help=text(
                "请输入肿瘤数目，编码方式需与原始建模数据一致。",
                "Enter the number of tumors. The coding should be consistent with the original modeling dataset."
            )
        )

        Tumor_diameter = st.number_input(
            text("最大肿瘤直径", "Tumor diameter"),
            min_value=0.0,
            max_value=30.0,
            value=5.0,
            step=0.1,
            help=text(
                "请输入最大肿瘤直径，单位通常为cm，需与原始建模数据一致。",
                "Enter the maximum tumor diameter, usually in cm. The unit should be consistent with the original modeling dataset."
            )
        )

        calculated_up_to_seven_sum = Tumor_number + Tumor_diameter

        st.caption(
            text(
                f"根据当前输入：肿瘤数目 + 最大肿瘤直径 = {calculated_up_to_seven_sum:.1f}。若≤7，通常符合Up-to-seven标准。",
                f"Based on current inputs: tumor number + maximum tumor diameter = {calculated_up_to_seven_sum:.1f}. A value ≤7 is generally within the up-to-seven criteria."
            )
        )

        vp_options = [0, 1, 2, 3, 4]

        vp_label_cn = {
            0: "0 = Vp0：无门静脉癌栓",
            1: "1 = Vp1：癌栓位于门静脉二级分支以远",
            2: "2 = Vp2：癌栓累及门静脉二级分支",
            3: "3 = Vp3：癌栓累及门静脉一级分支",
            4: "4 = Vp4：癌栓累及门静脉主干或对侧门静脉分支"
        }

        vp_label_en = {
            0: "0 = Vp0: no portal vein tumor thrombus",
            1: "1 = Vp1: thrombus distal to second-order branches",
            2: "2 = Vp2: thrombus involving second-order branches",
            3: "3 = Vp3: thrombus involving first-order branches",
            4: "4 = Vp4: thrombus involving main portal vein trunk or contralateral portal vein branch"
        }

        Vp_Classification = st.selectbox(
            text("门静脉癌栓分型", "Vp classification"),
            options=vp_options,
            index=0,
            format_func=lambda x: vp_label_cn[x] if IS_CN else vp_label_en[x],
            help=text(
                "请选择门静脉癌栓Vp分型，编码方式需与原始建模数据一致。",
                "Select the Vp classification code. The coding should be consistent with the original modeling dataset."
            )
        )

        uts_options = [0, 1]

        uts_label_cn = {
            0: "0 = 不符合Up-to-seven标准（最大肿瘤直径cm + 肿瘤数目 > 7）",
            1: "1 = 符合Up-to-seven标准（最大肿瘤直径cm + 肿瘤数目 ≤ 7）"
        }

        uts_label_en = {
            0: "0 = Beyond up-to-seven criteria (largest tumor diameter in cm + tumor number > 7)",
            1: "1 = Within up-to-seven criteria (largest tumor diameter in cm + tumor number ≤ 7)"
        }

        up_to_seven = st.selectbox(
            text("Up-to-seven标准", "Up-to-seven criteria"),
            options=uts_options,
            index=1 if calculated_up_to_seven_sum <= 7 else 0,
            format_func=lambda x: uts_label_cn[x] if IS_CN else uts_label_en[x],
            help=text(
                "Up-to-seven标准通常定义为：最大肿瘤直径(cm) + 肿瘤数目 ≤ 7。请选择与原始建模数据一致的0/1编码。",
                "The up-to-seven criteria are generally defined as: largest tumor diameter (cm) + number of tumors ≤ 7. Select the 0/1 code consistent with the original modeling dataset."
            )
        )

    with col2:

        BMI = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=50.0,
            value=23.0,
            step=0.1,
            help=text(
                "请输入体重指数。",
                "Enter body mass index."
            )
        )

        ALT = st.number_input(
            "ALT",
            min_value=0.0,
            max_value=1000.0,
            value=40.0,
            step=1.0,
            help=text(
                "请输入丙氨酸氨基转移酶水平。",
                "Enter alanine aminotransferase level."
            )
        )

        AST = st.number_input(
            "AST",
            min_value=0.0,
            max_value=1000.0,
            value=40.0,
            step=1.0,
            help=text(
                "请输入天冬氨酸氨基转移酶水平。",
                "Enter aspartate aminotransferase level."
            )
        )

        INR = st.number_input(
            "INR",
            min_value=0.5,
            max_value=5.0,
            value=1.0,
            step=0.01,
            help=text(
                "请输入国际标准化比值。",
                "Enter international normalized ratio."
            )
        )

    submitted = st.form_submit_button(
        text("计算预测概率", "Calculate prediction score")
    )


# ============================================================
# 7. 预测结果
# ============================================================

if submitted:

    input_data = {
        "Tumor_number": Tumor_number,
        "Vp_Classification": Vp_Classification,
        "up_to_seven": up_to_seven,
        "Tumor_diameter": Tumor_diameter,
        "BMI": BMI,
        "ALT": ALT,
        "AST": AST,
        "INR": INR
    }

    missing_input = [v for v in feature_vars if v not in input_data]

    if missing_input:
        st.error(
            text(
                f"网页输入变量缺少模型所需变量：{missing_input}",
                f"The web input is missing variables required by the model: {missing_input}"
            )
        )
        st.stop()

    input_data_ordered = {var: input_data[var] for var in feature_vars}

    X = pd.DataFrame([input_data_ordered], columns=feature_vars)

    pred_score = get_probability_of_class_1(model, X)
    score_group = assign_score_group(pred_score, thresholds)
    interpretation = get_group_interpretation(score_group)

    st.divider()

    st.subheader(
        text(
            "预测结果",
            "Prediction result"
        )
    )

    st.metric(
        label=text(
            "客观缓解预测概率",
            "Predicted probability of objective response"
        ),
        value=f"{pred_score * 100:.1f}%"
    )

    st.caption(
        text(
            f"模型原始预测评分：{pred_score:.3f}",
            f"Raw model prediction score: {pred_score:.3f}"
        )
    )

    st.metric(
        label=text(
            "预测评分分组",
            "Prediction-score group"
        ),
        value=score_group
    )

    if score_group == "Q4":
        st.success(interpretation)
    elif score_group == "Q1":
        st.warning(interpretation)
    else:
        st.info(interpretation)

    st.markdown(
        text(
            "### 输入变量汇总",
            "### Input summary"
        )
    )

    st.dataframe(
        pd.DataFrame([input_data_ordered]),
        use_container_width=True
    )


# ============================================================
# 8. 模型信息
# ============================================================

st.divider()

with st.expander(
    text(
        "模型信息 / 使用说明",
        "Model information / Instructions"
    )
):

    st.write(
        text(
            "模型类型：CatBoost分类模型",
            "Model type: CatBoost classifier"
        )
    )

    st.write(
        text(
            "结局定义：CR/PR = 1，SD/PD = 0",
            "Outcome definition: CR/PR = 1, SD/PD = 0"
        )
    )

    st.write(
        text(
            "模型使用变量：",
            "Variables used by the model:"
        )
    )

    st.write(feature_vars)

    st.write(
        text(
            "基于训练集预测评分生成的四分位截点：",
            "Quartile cutoffs of prediction score derived from the training set:"
        )
    )

    st.json(thresholds)

    st.markdown(
        text(
            """
**注意：**

该工具仅用于科研展示和辅助评估，不能替代临床医生判断或多学科讨论决策。

预测结果应结合患者整体病情、影像学特征、肝功能状态以及多学科讨论综合判断。
""",
            """
**Note:**

This tool is intended for research demonstration and auxiliary assessment only. It should not replace clinical judgment or multidisciplinary decision-making.

The prediction result should be interpreted together with the patient’s overall clinical condition, imaging features, liver function, and multidisciplinary discussion.
"""
        )
    )


st.caption(
    text(
        "TAE-HAIC客观缓解预测模型 | 仅供科研使用",
        "TAE-HAIC Objective Response Prediction Model | Research use only"
    )
)
