# 3DFaceTexGen

## 프로젝트 목적
- 2D 입력 이미지(인물 사진)로부터 UV Texture를 추출할 때 발생하는 self-occlusion 문제를 개선하기 위함<br/>

<img src="https://user-images.githubusercontent.com/102565074/229715821-b90aa79e-15cc-4093-ae5f-2d7cd97ad5b2.png" width="300" height="300"/>

## 프로젝트 내용
- 개요<br/>

![개요](https://user-images.githubusercontent.com/102565074/229708778-15eb0f89-0eda-406e-b6ee-9d611b2cc6f4.png)

- 설명
1) **Inversion**<br/>
2) **Editing**<br/>
2-1. Yaw [-30] :arrow_right: 생성 좌 I<br/>
2-2. Yaw [+30] :arrow_right: 생성 우 I<br/>
3) **Unwrapping**<br/>
3-1. 생성 좌 I :arrow_right: 생성 좌 T<br/>
3-2. 생성 우 I :arrow_right: 생성 우 T<br/>
3-3. 입력 I :arrow_right: 입력 T<br/>
4) **Blending**<br/>
4-1. 생성 우 T + 생성 좌 T + 마스크 좌 :arrow_right: 기준 T<br/>
4-2. 기준 T + 입력 T + 마스크 중 :arrow_right: 결과 T<br/>

## 프로젝트 결과
|Before|After|
|---|---|
|<img src="https://user-images.githubusercontent.com/102565074/229714380-f12f2350-a356-44e0-a97f-ec2d4274895a.png" width="300" height="300"/>|<img src="https://user-images.githubusercontent.com/102565074/229716471-488b4b26-f5d8-4f9b-b7dc-fa01d48d4947.png" width="300" height="300"/>|
|<img src="https://user-images.githubusercontent.com/102565074/229714389-e1cb75be-2d59-4a6c-82f1-178c6ff6dc58.png" width="300" height="300"/>|<img src="https://user-images.githubusercontent.com/102565074/229716483-8ae1c652-a964-44b3-b5e3-40fcee787695.png" width="300" height="300"/>|
