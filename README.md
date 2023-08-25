# CongestionAI

- 엣지 디바이스에서 처리 가능한 경량화된 Crowd Counting AI 개발
- 경량화를 위하여 Teacher Model (P2PNet)의 지식을 Student Model (MCNN)에 증류 (Knowledge Distillation)

## 사용할 데이터셋
- ShanghaiDatasetA: https://paperswithcode.com/dataset/shanghaitech

## 진행 사항
- Teacher 모델 선정: SOTA 모델인 P2PNet으로 선정 <span style='background-color: #dcffe4'>(~2023.07.05, 완료)</span>
- P2PNet 전처리 및 증강 추가 <span style='background-color: #dcffe4'>(~2023.07.08, 완료)</span>
- P2PNet 학습 <span style='background-color: #dcffe4'>(~2023.07.15, 완료)</span>
- 최적의 P2PNet Weight 선택 <span style='background-color: #dcffe4'>(~2023.07.20, 완료)</span>
- P2PNet Distillation loss 개발 <span style='background-color: #dcffe4'>(~2023.07.25, 완료)</span>
- 기존 MCNN 학습 및 성능 측정 <span style='background-color: #dcffe4'>(~2023.08.10, 완료)</span>
- MCNN 모델 성능 향상을 위한 전처리 및 증강 추가 <span style='background-color: #dcffe4'>(~2023.08.25, 완료)</span>
- MCNN 전처리 적용 학습 및 성능 측정 <span style='background-color: #F7DDBE'>(~2023.08.27, 진행 중)</span>
- P2PNet 사용 MCNN 지식 증류(Knowledge Distillation) 적용 <span style='background-color: #ffdce0'>(~2023.09.05, 진행 예정)</span>

## original codes
- MCNN: https://github.com/CommissarMa/MCNN-pytorch
- transforms: https://github.com/pxq0312/SFANet-crowd-counting/blob/master/transforms.py