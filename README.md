# tmp_test_seethough

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/tmp_test_seethough/blob/main/seethrough_cel.ipynb)

NAI(또는 아무 플랫 컬러 애니 일러스트) 한 장을 **부위별로 나뉜 벡터 캐릭터**로 만들고, GSAP으로 움직이는 셀 애니메이션 HTML까지 뽑는 Colab 파이프라인.

```
원본 PNG ─► See-through (레이어 분리 + 가려진 부분 채우기, NF4 저VRAM)
          ─► 레이어마다 업스케일 (tmp_upscaler의 spandrel 모델, 알파 따로 처리)
          ─► 레이어마다 SVG 변환 (팔레트 + 디테일 보존 + 소프트 그라데이션)
          ─► character.svg + character.html (숨쉬기 / 고개 / 머리카락 / 깜빡임)
```

## 사용법 (Colab)
1. 위 **Open In Colab** 배지로 열고 GPU 런타임 선택
   (배지는 GitHub `main` 브랜치에서 노트북을 바로 불러오고, 설치 셀이 매번 `git pull`로 코드도 최신으로 맞춥니다.
   Colab에서 "사본 저장"한 노트북은 그 시점에 고정되니 항상 배지로 여세요)
2. 설치 셀 실행 → **세션 다시 시작** 한 번
3. 원본 업로드 (선택: 같은 그림의 눈 감은 버전) → 실행 → 미리보기 → zip 다운로드

## 구성
| 파일 | 역할 |
|---|---|
| `pipeline/decompose.py` | See-through 실행, PSD를 레이어 배열로 읽기 (뒤→앞 순서) |
| `pipeline/upscale.py` | tmp_upscaler `server_core`로 RGBA 업스케일 (알파 블리드 + 알파 별도 업스케일) |
| `pipeline/vectorize.py` | 투명도 인식 SVG 변환 |
| `pipeline/assemble.py` | 레이어 역할 분류, SVG 조립, GSAP HTML 템플릿 |
| `pipeline/run.py` | 전체 실행 CLI (`python -m pipeline.run IMAGE ...`) |
| `setup_colab.sh` | See-through / tmp_upscaler 클론 + 필요한 패키지만 설치 |
| `build_notebook.py` | 노트북 생성기 (노트북은 손으로 고치지 말고 이걸 고친 뒤 재생성) |

## 옵션
- `--from-psd X.psd`: See-through 건너뛰고 기존 PSD 사용
- `--no-upscale`: 업스케일 생략 (빠른 테스트)
- `--model`, `--tile`: 업스케일 모델 / 타일 크기 (VRAM 부족 시 256)
- `--colors --merge --detail --speckle --soft`: 벡터화 세부값

## 애니메이션 규칙
레이어 이름(See-through 태그)으로 역할을 정합니다: `front hair`/`back hair` → 흔들림, 눈 관련 → 깜빡임, 머리 관련 → 고개 움직임(공통 기준점), 전체 → 숨쉬기.
눈 감은 그림을 같이 주면 눈 레이어를 **셀처럼 바꿔 끼웁니다**. 없으면 눈을 세로로 눌러서 임시 표현합니다.
움직임 크기/속도는 `assemble.DEFAULT_PARAMS`에서 조절.

## 테스트 상태
- 검증됨: PSD 읽기 → SVG 변환 → 조립 → HTML 애니메이션 (가짜 PSD로 헤드리스 브라우저에서 확인, JS 오류 없음)
- 미검증: Colab에서 See-through 설치/실행, GPU 업스케일 (GPU 없는 환경에서 작성)
