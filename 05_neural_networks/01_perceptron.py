"""
01. 퍼셉트론 (Perceptron)
신경망의 가장 기본적인 단위인 퍼셉트론을 처음부터 구현해봅니다.
"""

import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    """단일 퍼셉트론 클래스"""
    
    def __init__(self, input_size, learning_rate=0.01):
        """
        퍼셉트론 초기화
        
        Args:
            input_size: 입력 특성의 개수
            learning_rate: 학습률
        """
        self.learning_rate = learning_rate
        # 가중치를 작은 랜덤 값으로 초기화
        self.weights = np.random.randn(input_size) * 0.01
        self.bias = 0.0
        
        # 학습 과정 기록을 위한 변수들
        self.errors = []
        self.weights_history = []
    
    def activation_function(self, z):
        """활성화 함수 (계단 함수)"""
        return 1 if z >= 0 else 0
    
    def predict(self, inputs):
        """예측 수행"""
        # 가중합 계산: w·x + b
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return self.activation_function(weighted_sum)
    
    def train(self, training_inputs, labels, epochs=100):
        """퍼셉트론 학습"""
        print(f"퍼셉트론 학습 시작 (에포크: {epochs})")
        
        for epoch in range(epochs):
            total_error = 0
            
            # 가중치 기록
            self.weights_history.append(self.weights.copy())
            
            for i in range(len(training_inputs)):
                # 예측
                prediction = self.predict(training_inputs[i])
                
                # 오차 계산
                error = labels[i] - prediction
                total_error += abs(error)
                
                # 가중치 업데이트 (퍼셉트론 학습 규칙)
                self.weights += self.learning_rate * error * training_inputs[i]
                self.bias += self.learning_rate * error
            
            self.errors.append(total_error)
            
            # 진행 상황 출력
            if epoch % 20 == 0 or epoch == epochs - 1:
                accuracy = (len(training_inputs) - total_error) / len(training_inputs) * 100
                print(f"에포크 {epoch:3d}: 오차 {total_error}, 정확도 {accuracy:.1f}%")
            
            # 수렴 조건: 오차가 0이면 학습 완료
            if total_error == 0:
                print(f"에포크 {epoch}에서 완벽하게 수렴!")
                break

def create_and_gate_data():
    """AND 게이트 데이터 생성"""
    inputs = np.array([
        [0, 0],
        [0, 1], 
        [1, 0],
        [1, 1]
    ])
    
    labels = np.array([0, 0, 0, 1])  # AND 게이트 진리표
    
    return inputs, labels

def create_or_gate_data():
    """OR 게이트 데이터 생성"""
    inputs = np.array([
        [0, 0],
        [0, 1],
        [1, 0], 
        [1, 1]
    ])
    
    labels = np.array([0, 1, 1, 1])  # OR 게이트 진리표
    
    return inputs, labels

def create_linearly_separable_data():
    """선형 분리 가능한 2D 데이터 생성"""
    np.random.seed(42)
    
    # 클래스 0 데이터 (왼쪽 하단)
    class_0 = np.random.randn(20, 2) * 0.5 + np.array([0, 0])
    
    # 클래스 1 데이터 (오른쪽 상단)
    class_1 = np.random.randn(20, 2) * 0.5 + np.array([2, 2])
    
    # 데이터 합치기
    inputs = np.vstack([class_0, class_1])
    labels = np.hstack([np.zeros(20), np.ones(20)])
    
    return inputs, labels

def visualize_decision_boundary(perceptron, inputs, labels, title="Decision Boundary"):
    """결정 경계 시각화"""
    try:
        plt.figure(figsize=(10, 8))
        
        # 데이터 포인트 그리기
        class_0_mask = labels == 0
        class_1_mask = labels == 1
        
        plt.scatter(inputs[class_0_mask, 0], inputs[class_0_mask, 1], 
                   c='red', marker='o', s=100, alpha=0.7, label='클래스 0')
        plt.scatter(inputs[class_1_mask, 0], inputs[class_1_mask, 1], 
                   c='blue', marker='s', s=100, alpha=0.7, label='클래스 1')
        
        # 결정 경계 그리기
        if len(perceptron.weights) == 2:  # 2D 데이터인 경우
            x_min, x_max = inputs[:, 0].min() - 0.5, inputs[:, 0].max() + 0.5
            
            # 결정 경계: w1*x1 + w2*x2 + b = 0
            # x2 = -(w1*x1 + b) / w2
            if perceptron.weights[1] != 0:
                x1_line = np.array([x_min, x_max])
                x2_line = -(perceptron.weights[0] * x1_line + perceptron.bias) / perceptron.weights[1]
                plt.plot(x1_line, x2_line, 'g-', linewidth=2, label='결정 경계')
        
        plt.xlabel('특성 1')
        plt.ylabel('특성 2')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
    except ImportError:
        print("Matplotlib이 설치되지 않아 시각화를 건너뜁니다.")
    except Exception as e:
        print(f"시각화 중 오류 발생: {e}")

def demo_and_gate():
    """AND 게이트 학습 데모"""
    print("=" * 50)
    print("AND 게이트 학습 데모")
    print("=" * 50)
    
    # 데이터 준비
    inputs, labels = create_and_gate_data()
    
    print("AND 게이트 진리표:")
    for i in range(len(inputs)):
        print(f"입력: {inputs[i]} → 출력: {labels[i]}")
    
    # 퍼셉트론 생성 및 학습
    perceptron = Perceptron(input_size=2, learning_rate=0.1)
    perceptron.train(inputs, labels, epochs=100)
    
    # 학습된 가중치 출력
    print(f"\n학습된 가중치: {perceptron.weights}")
    print(f"편향: {perceptron.bias:.3f}")
    
    # 테스트
    print("\n학습 결과 테스트:")
    for i in range(len(inputs)):
        prediction = perceptron.predict(inputs[i])
        print(f"입력: {inputs[i]} → 예측: {prediction}, 정답: {labels[i]}")

def demo_or_gate():
    """OR 게이트 학습 데모"""
    print("\n" + "=" * 50)
    print("OR 게이트 학습 데모")
    print("=" * 50)
    
    # 데이터 준비
    inputs, labels = create_or_gate_data()
    
    print("OR 게이트 진리표:")
    for i in range(len(inputs)):
        print(f"입력: {inputs[i]} → 출력: {labels[i]}")
    
    # 퍼셉트론 생성 및 학습
    perceptron = Perceptron(input_size=2, learning_rate=0.1)
    perceptron.train(inputs, labels, epochs=100)
    
    # 학습된 가중치 출력
    print(f"\n학습된 가중치: {perceptron.weights}")
    print(f"편향: {perceptron.bias:.3f}")
    
    # 테스트
    print("\n학습 결과 테스트:")
    for i in range(len(inputs)):
        prediction = perceptron.predict(inputs[i])
        print(f"입력: {inputs[i]} → 예측: {prediction}, 정답: {labels[i]}")

def demo_2d_classification():
    """2D 분류 문제 데모"""
    print("\n" + "=" * 50)
    print("2D 분류 문제 데모")
    print("=" * 50)
    
    # 데이터 준비
    inputs, labels = create_linearly_separable_data()
    
    print(f"데이터 개수: {len(inputs)}개")
    print(f"클래스 0: {np.sum(labels == 0)}개")
    print(f"클래스 1: {np.sum(labels == 1)}개")
    
    # 퍼셉트론 생성 및 학습
    perceptron = Perceptron(input_size=2, learning_rate=0.01)
    perceptron.train(inputs, labels, epochs=1000)
    
    # 최종 정확도 계산
    correct = 0
    for i in range(len(inputs)):
        prediction = perceptron.predict(inputs[i])
        if prediction == labels[i]:
            correct += 1
    
    accuracy = correct / len(inputs) * 100
    print(f"\n최종 정확도: {accuracy:.1f}%")
    
    # 결정 경계 시각화
    visualize_decision_boundary(perceptron, inputs, labels, "2D 분류 결과")
    
    return perceptron, inputs, labels

def analyze_learning_process(perceptron):
    """학습 과정 분석"""
    print("\n" + "=" * 50)
    print("학습 과정 분석")
    print("=" * 50)
    
    print(f"총 에포크 수: {len(perceptron.errors)}")
    print(f"초기 오차: {perceptron.errors[0]}")
    print(f"최종 오차: {perceptron.errors[-1]}")
    
    # 학습 곡선 그리기
    try:
        plt.figure(figsize=(12, 4))
        
        # 오차 곡선
        plt.subplot(1, 2, 1)
        plt.plot(perceptron.errors)
        plt.title('학습 곡선 (오차)')
        plt.xlabel('에포크')
        plt.ylabel('총 오차')
        plt.grid(True, alpha=0.3)
        
        # 가중치 변화
        if perceptron.weights_history:
            plt.subplot(1, 2, 2)
            weights_array = np.array(perceptron.weights_history)
            for i in range(weights_array.shape[1]):
                plt.plot(weights_array[:, i], label=f'가중치 {i+1}')
            plt.title('가중치 변화')
            plt.xlabel('에포크')
            plt.ylabel('가중치 값')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib이 설치되지 않아 그래프를 건너뜁니다.")
    except Exception as e:
        print(f"그래프 생성 중 오류: {e}")

def demonstrate_xor_limitation():
    """XOR 문제를 통한 퍼셉트론의 한계 데모"""
    print("\n" + "=" * 50)
    print("퍼셉트론의 한계: XOR 문제")
    print("=" * 50)
    
    # XOR 데이터
    inputs = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    labels = np.array([0, 1, 1, 0])  # XOR 진리표
    
    print("XOR 게이트 진리표:")
    for i in range(len(inputs)):
        print(f"입력: {inputs[i]} → 출력: {labels[i]}")
    
    # 퍼셉트론으로 XOR 학습 시도
    perceptron = Perceptron(input_size=2, learning_rate=0.1)
    perceptron.train(inputs, labels, epochs=1000)
    
    # 테스트
    print("\n학습 결과 (실패 예상):")
    correct = 0
    for i in range(len(inputs)):
        prediction = perceptron.predict(inputs[i])
        is_correct = prediction == labels[i]
        correct += is_correct
        print(f"입력: {inputs[i]} → 예측: {prediction}, 정답: {labels[i]} {'✓' if is_correct else '✗'}")
    
    accuracy = correct / len(inputs) * 100
    print(f"\n정확도: {accuracy:.1f}%")
    print("\n분석: XOR는 선형 분리가 불가능하므로 단일 퍼셉트론으로는 해결할 수 없습니다.")
    print("해결책: 다층 퍼셉트론(MLP)이나 비선형 활성화 함수 사용")

if __name__ == "__main__":
    print("퍼셉트론 - 신경망의 기초")
    print("=" * 50)
    
    # 기본 논리 게이트 학습
    demo_and_gate()
    demo_or_gate()
    
    # 2D 분류 문제
    perceptron, inputs, labels = demo_2d_classification()
    
    # 학습 과정 분석
    analyze_learning_process(perceptron)
    
    # 퍼셉트론의 한계 데모
    demonstrate_xor_limitation()
    
    print("\n" + "=" * 50)
    print("퍼셉트론 학습 완료!")
    print("핵심 개념:")
    print("- 가중합과 활성화 함수의 역할")
    print("- 퍼셉트론 학습 규칙")
    print("- 선형 분리 가능성의 중요성")
    print("- 단일 퍼셉트론의 한계 (XOR 문제)")
    print("\n다음 단계: 다층 퍼셉트론으로 XOR 문제 해결하기")