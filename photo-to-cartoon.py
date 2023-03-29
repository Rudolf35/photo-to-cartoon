import cv2
import numpy as np


def kmeansColorCluster(image, clusters, rounds):
        """
        Parameters
            image <np.ndarray> : 이미지
            clusters <int> : 클러스터 개수 (군집화 개수)
            rounds <int> : 알고리즘을 몇 번 실행할지 (보통 1)
        returns
            clustered Image <np.ndarray> : 결과 이미지
            SSE <float> : 오차 제곱 합
        """
        
        height, width = image.shape[:2]
        samples = np.zeros([ height * width, 3 ], dtype=np.float32)
        
        count = 0
        for x in range(height):
            for y in range(width):
                samples[count] = image[x][y]
                count += 1
        
        '''
        # compactness : SSE = 오차 제곱 합
        # labels : 레이블 배열 (0과 1로 표현)
        # centers : 클러스터 중심 좌표 (k개로 군집화된 색상들)
        '''
        compactness, labels, centers = cv2.kmeans(
                    samples, # 비지도 학습 데이터 정렬
                    clusters, # 군집화 개수
                    None, # 각 샘플의 군집 번호 정렬
                    # criteria : kmeans 알고리즘 반복 종료 기준 설정
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 
                                10000, # max_iter 
                                0.0001), # epsilon 
                    # attempts : 다른 초기 중앙값을 이용해 반복 실행할 횟수
                    attempts = rounds, 
                    # flags : 초기 중앙값 설정 방법
                    flags = cv2.KMEANS_PP_CENTERS)
        
        centers = np.uint8(centers)
        res = centers[labels.flatten()]
        
        # 결과 이미지, 초기 중앙값, 오차제곱합 반환
        return res.reshape((image.shape)), centers, round(compactness, 4)

# 이미지를 불러옵니다.
img = cv2.imread('image.jpg')
# 이미지를 회색조로 변경합니다.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 가우시안 블러(흐리게) 처리합니다.
gray = cv2.medianBlur(gray, 5)
# 엣지 검출을 수행합니다.
edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
# 컬러 이미지로 변경합니다.
color = cv2.bilateralFilter(img, 9, 300, 300)
# BGR 색공간을 RGB로 변환
color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
# 24개 색상으로 이미지 군집화
cluster, _, _ = kmeansColorCluster(color, 24, 1)
# RGB 색공간을 BGR로 변환
cluster = cv2.cvtColor(cluster, cv2.COLOR_RGB2BGR)
# 컬러 이미지와 엣지를 합성합니다.
cartoon = cv2.bitwise_and(cluster, cluster, mask=edges)

# 결과를 출력합니다.
cv2.imshow("Cartoon", cartoon)
cv2.waitKey(0)
cv2.destroyAllWindows()
