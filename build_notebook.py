"""Generate seethrough_cel.ipynb (keep logic in pipeline/, the notebook only drives it)."""
import json
from pathlib import Path


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


cells = [
    md("# NAI → See-through → 업스케일 → SVG → 셀 애니메이션\n"
       "런타임 > 런타임 유형 변경 > **GPU** 선택 후 위에서부터 차례로 실행하세요.\n\n"
       "결과: 레이어별 SVG, 합친 `character.svg`, 움직이는 `character.html`, 미리보기 PNG를 zip으로 받습니다."),
    code("!nvidia-smi --query-gpu=name,memory.total --format=csv"),
    md("## 1. 설치 (처음 한 번)\n설치가 끝나면 **런타임 > 세션 다시 시작**을 한 번 해 주세요 (numpy 등 버전이 바뀌어서 필요합니다)."),
    code("REPO = 'https://github.com/HisameOgasahara/tmp_test_seethough'\n"
         "import os\n"
         "if not os.path.exists('/content/tmp_test_seethough'):\n"
         "    !git clone --depth 1 {REPO} /content/tmp_test_seethough\n"
         "else:\n"
         "    !git -C /content/tmp_test_seethough pull -q\n"
         "!bash /content/tmp_test_seethough/setup_colab.sh /content"),
    md("## 2. 이미지 올리기\n먼저 **원본 그림**을 올리고, 이어서 나오는 창에는 **눈 감은 버전**(NAI 인페인팅)을 올리세요. 눈 감은 버전이 없으면 두 번째 창은 취소하면 됩니다 (그 경우 깜빡임은 임시로 눈을 납작하게 눌러서 표현)."),
    code("import os\nfrom google.colab import files\n"
         "os.chdir('/content')  # files.upload() saves into the current folder\n"
         "MAIN = os.path.abspath(list(files.upload())[0])\n"
         "try:\n"
         "    CLOSED = os.path.abspath(list(files.upload())[0])\n"
         "except Exception:  # cancelled / nothing chosen\n"
         "    CLOSED = None\n"
         "print('main:', MAIN, '| closed eyes:', CLOSED)"),
    md("## 3. 실행\n- `MODEL`: `4x-UltraSharpV2_Lite` 또는 `2x-AnimeSharpV4_Fast_RCAN_PU`\n"
       "- `FULL_PRECISION=True`는 VRAM 16GB 이상(A100/L4)에서만. T4는 기본값(NF4) 그대로\n"
       "- 이미 See-through PSD가 있으면 `FROM_PSD`에 경로를 넣으면 분리 단계를 건너뜁니다"),
    code("import sys; sys.path.insert(0, '/content/tmp_test_seethough')\n"
         "from pipeline import run\n\n"
         "MODEL = '4x-UltraSharpV2_Lite'\n"
         "FULL_PRECISION = False\n"
         "FROM_PSD = None\n"
         "OUT = '/content/out'\n\n"
         "argv = [MAIN, '--out', OUT, '--seethrough', '/content/see-through', '--upscaler', '/content/tmp_upscaler',\n"
         "        '--model', MODEL]\n"
         "if CLOSED: argv += ['--closed-eyes', CLOSED]\n"
         "if FULL_PRECISION: argv += ['--full-precision']\n"
         "if FROM_PSD: argv += ['--from-psd', FROM_PSD]\n"
         "ZIP = run.main(argv)"),
    md("## 4. 미리보기"),
    code("from IPython.display import Image, HTML, display\n"
         "display(Image(OUT + '/preview.png', width=480))\n"
         "display(HTML(open(OUT + '/character.html', encoding='utf-8').read()))"),
    md("## 5. 다운로드"),
    code("from google.colab import files\nfiles.download(ZIP)"),
]

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
for c in nb["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)
Path(__file__).with_name("seethrough_cel.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote seethrough_cel.ipynb")
