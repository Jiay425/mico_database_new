from flask import Flask, request, jsonify
import time
import random

app = Flask(__name__)

# 定义预测接口
@app.route('/predict', methods=['POST'])
def predict():
    # 1. 获取 Java 传来的数据
    data = request.get_json()
    patient_id = data.get('patient_id')

    print(f"【Python AI】收到预测请求，病人ID: {patient_id}")

    # 2. 模拟加载模型和计算 (这里替换为您真实的 XGBoost/PyTorch 代码)
    print("【Python AI】正在加载模型... 计算特征权重...")
    time.sleep(2) # 模拟计算耗时

    # 3. 模拟生成结果
    risk_score = round(random.uniform(0, 1), 4) # 随机生成 0-1 之间的风险值
    result = {
        "patient_id": patient_id,
        "risk_score": risk_score,
        "risk_level": "HIGH" if risk_score > 0.7 else "LOW",
        "suggestion": "建议进行肠道菌群进一步测序" if risk_score > 0.7 else "健康状况良好"
    }

    print(f"【Python AI】计算完成: {result}")
    return jsonify(result)

if __name__ == '__main__':
    # 启动在 5001 端口，避免和 Spring Boot (5000) 冲突
    app.run(host='0.0.0.0', port=5001)