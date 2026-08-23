# -*- coding: utf-8 -*-
"""캡처한 실제 에이전트 출력(out_agent_capture.json)을 판독 화면 HTML로 렌더."""
import json, html, re, io, datetime, sys

SRC = sys.argv[1] if len(sys.argv) > 1 else 'out_agent_capture.json'
DST = sys.argv[2] if len(sys.argv) > 2 else 'agent_output.html'

D = json.load(open(SRC, encoding='utf-8'))
CASE = D['cases']['분뇨제거']
ALT = D['cases']['액비살포']
E = html.escape


def hhmm(iso):
    s = str(iso)
    try:
        return datetime.datetime.fromisoformat(s).strftime('%m/%d %H:%M')
    except Exception:
        return s[:16]


def gcls(g):
    return {'낮음': 'g-low', '주의': 'g-mid', '위험': 'g-high'}.get(g, 'g-mid')


def md2html(t):
    out, in_ul = [], False
    for raw in (t or '').split('\n'):
        ln = raw.strip()
        if ln == '---':
            if in_ul:
                out.append('</ul>'); in_ul = False
            out.append('<hr class="soft">'); continue
        if not ln:
            if in_ul:
                out.append('</ul>'); in_ul = False
            continue
        b = E(ln)
        b = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', b)
        m = re.match(r'^(#{2,4})\s+(.*)', b)
        if m:
            if in_ul:
                out.append('</ul>'); in_ul = False
            out.append('<h4 class="gh">' + m.group(2) + '</h4>'); continue
        if b.startswith('&gt; '):
            if in_ul:
                out.append('</ul>'); in_ul = False
            out.append('<p class="quote">' + b[5:] + '</p>'); continue
        if b.startswith('* ') or b.startswith('- '):
            if not in_ul:
                out.append('<ul class="gl">'); in_ul = True
            out.append('<li>' + b[2:] + '</li>'); continue
        if in_ul:
            out.append('</ul>'); in_ul = False
        out.append('<p>' + b + '</p>')
    if in_ul:
        out.append('</ul>')
    return '\n'.join(out)


def windows(items):
    rows = []
    for i, w in enumerate(items, 1):
        why = E(w['reasons'][1]) if len(w['reasons']) > 1 else ''
        rows.append(
            '<div class="win ' + gcls(w['grade']) + '">'
            '<div class="win-rank">' + str(i) + '</div>'
            '<div class="win-main">'
            '<div class="win-time">' + E(hhmm(w['start'])) +
            ' <span class="arrow">&rarr;</span> ' + E(hhmm(w['end'])) + '</div>'
            '<div class="win-why">' + why + '</div></div>'
            '<div class="win-nums">'
            '<div><span class="lab">6시간 위험</span><span class="val">' + str(w['window_risk']) + '</span></div>'
            '<div><span class="lab">비교점수</span><span class="val">' + str(w['recommendation_score']) + '</span></div>'
            '</div>'
            '<div><span class="chip ' + gcls(w['grade']) + '">' + E(w['grade']) + '</span></div>'
            '</div>')
    return '\n'.join(rows)


def evidence(items):
    out = []
    for e in items:
        snip = ' '.join(str(e.get('snippet', '')).split())[:230]
        manual = e.get('doc_type') == 'manual'
        typ = '실무' if manual else '법령'
        out.append(
            '<article class="ev ' + ('ev-m' if manual else 'ev-l') + '">'
            '<div class="ev-top"><span class="ev-type">' + typ + '</span>'
            '<span class="ev-src">' + E(str(e.get('source_file', ''))[:46]) + '</span></div>'
            '<div class="ev-unit">' + E(str(e.get('unit') or '단원 미확인')) +
            ' <span class="ev-pg">p.' + str(e.get('page', '?')) + '</span></div>'
            '<p class="ev-snip">' + E(snip) + '&hellip;</p></article>')
    return '\n'.join(out)


def cal_rows(items, n=30):
    out = []
    for it in items[:n]:
        hr = '일 단위' if it.get('hour') is None else ('%02d시' % int(it['hour']))
        out.append(
            '<tr><td class="mono">' + E(str(it.get('date'))) + '</td>'
            '<td class="mono">' + hr + '</td>'
            '<td class="mono num">' + str(it.get('risk_score')) + '</td>'
            '<td><span class="chip sm ' + gcls(it.get('risk_grade')) + '">' +
            E(str(it.get('risk_grade'))) + '</span></td>'
            '<td class="mono dim">' + E(str(it.get('horizon'))) + '</td></tr>')
    return '\n'.join(out)


SRCS = CASE['source_statuses']


def srcpill(key, label):
    s = SRCS.get(key) or {}
    st = s.get('state', 'unknown')
    cls = {'connected': 'g-low', 'fixture': 'g-mid',
           'unavailable': 'g-high', 'stale': 'g-mid'}.get(st, 'g-mid')
    return ('<div class="stat"><span class="stat-k">' + E(label) + '</span>'
            '<span class="chip ' + cls + '">' + E(st) + '</span>'
            '<span class="stat-v">' + E(str(s.get('name', '—'))[:54]) + '</span></div>')


alt_rows = '\n'.join(
    '<tr><td>' + E(w['grade']) + '</td><td class="mono">' + E(hhmm(w['start'])) +
    '</td><td class="mono num">' + str(w['window_risk']) +
    '</td><td class="mono num">' + str(w['recommendation_score']) + '</td></tr>'
    for w in ALT['recommended'])

n_cal = len(D['calendar']['data']['items'])

CSS = """
:root{
 --ground:#F4F6F7; --surface:#fff; --surface-2:#EBEFF1;
 --ink:#141E25; --ink-2:#42565F; --muted:#6E838D;
 --line:#D2DBDF; --line-2:#B9C6CC;
 --accent:#1F6E7E; --accent-soft:#DCEBEE;
 --low:#3D7A4E; --low-s:#DFEDE3;
 --mid:#9A6C1F; --mid-s:#F4E9D2;
 --high:#A3524A; --high-s:#F4E0DD;
 --r:10px;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
 --ground:#0E1519; --surface:#151E23; --surface-2:#1C272D;
 --ink:#DFE8EC; --ink-2:#A9BAC2; --muted:#7D8F98;
 --line:#2A3841; --line-2:#3B4C56;
 --accent:#5CB3C4; --accent-soft:#132B31;
 --low:#6FB183; --low-s:#152618;
 --mid:#D3A458; --mid-s:#2A2113;
 --high:#D98B7F; --high-s:#2B1917;
}}
:root[data-theme="dark"]{
 --ground:#0E1519; --surface:#151E23; --surface-2:#1C272D;
 --ink:#DFE8EC; --ink-2:#A9BAC2; --muted:#7D8F98;
 --line:#2A3841; --line-2:#3B4C56;
 --accent:#5CB3C4; --accent-soft:#132B31;
 --low:#6FB183; --low-s:#152618;
 --mid:#D3A458; --mid-s:#2A2113;
 --high:#D98B7F; --high-s:#2B1917;
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
 font-family:"IBM Plex Sans KR",system-ui,"Malgun Gothic",sans-serif;
 font-size:16px;line-height:1.65;-webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:0 22px 90px}
.mono,code{font-family:"IBM Plex Mono",ui-monospace,monospace}
code{background:var(--surface-2);padding:.12em .38em;border-radius:4px;font-size:.87em}
.num{font-variant-numeric:tabular-nums}
.dim{color:var(--muted)}

header{border-bottom:1px solid var(--line);padding:58px 0 30px;margin-bottom:38px}
.eyebrow{font-family:"IBM Plex Mono",monospace;font-size:12px;font-weight:600;
 letter-spacing:.13em;text-transform:uppercase;color:var(--accent);margin:0 0 12px}
h1{font-size:clamp(28px,4vw,42px);line-height:1.15;font-weight:700;
 letter-spacing:-.02em;margin:0 0 14px;text-wrap:balance}
.lede{font-size:17.5px;color:var(--ink-2);margin:0;max-width:64ch}
.stamp{margin-top:20px;font-family:"IBM Plex Mono",monospace;font-size:12.5px;
 color:var(--muted);display:flex;flex-wrap:wrap;gap:6px 20px}

section{margin-top:54px}
h2{font-size:22px;font-weight:600;letter-spacing:-.014em;margin:0 0 6px;
 display:flex;align-items:baseline;gap:11px}
h2 .n{font-family:"IBM Plex Mono",monospace;font-size:13px;font-weight:600;
 color:var(--accent);letter-spacing:.06em}
.sub{color:var(--ink-2);margin:0 0 22px;max-width:66ch}
h3{font-size:16px;font-weight:600;margin:30px 0 12px}

.chip{font-family:"IBM Plex Mono",monospace;font-size:11px;font-weight:600;
 letter-spacing:.03em;padding:3px 9px;border-radius:999px;white-space:nowrap}
.chip.sm{font-size:10px;padding:2px 7px}
.g-low{background:var(--low-s);color:var(--low)}
.g-mid{background:var(--mid-s);color:var(--mid)}
.g-high{background:var(--high-s);color:var(--high)}

.stats{display:grid;gap:9px;grid-template-columns:repeat(auto-fit,minmax(310px,1fr))}
.stat{background:var(--surface);border:1px solid var(--line);border-radius:var(--r);
 padding:12px 15px;display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.stat-k{font-size:14px;font-weight:600;min-width:74px}
.stat-v{font-family:"IBM Plex Mono",monospace;font-size:11.5px;color:var(--muted);
 flex:1;word-break:break-all}

.win{background:var(--surface);border:1px solid var(--line);border-left:4px solid var(--line-2);
 border-radius:0 var(--r) var(--r) 0;padding:14px 18px;margin-bottom:9px;
 display:flex;align-items:center;gap:18px;flex-wrap:wrap}
.win.g-low{border-left-color:var(--low)}
.win.g-mid{border-left-color:var(--mid)}
.win.g-high{border-left-color:var(--high)}
.win-rank{font-family:"IBM Plex Mono",monospace;font-size:19px;font-weight:600;
 color:var(--muted);min-width:20px}
.win-main{flex:1;min-width:210px}
.win-time{font-family:"IBM Plex Mono",monospace;font-size:16px;font-weight:600;letter-spacing:-.01em}
.win-time .arrow{color:var(--muted);margin:0 4px}
.win-why{font-size:12.5px;color:var(--muted);margin-top:3px}
.win-nums{display:flex;gap:20px}
.win-nums div{display:flex;flex-direction:column}
.win-nums .lab{font-family:"IBM Plex Mono",monospace;font-size:10px;letter-spacing:.06em;
 color:var(--muted);text-transform:uppercase}
.win-nums .val{font-family:"IBM Plex Mono",monospace;font-size:15px;font-weight:600;
 font-variant-numeric:tabular-nums}

.evs{display:grid;gap:11px;grid-template-columns:repeat(auto-fit,minmax(320px,1fr))}
.ev{background:var(--surface);border:1px solid var(--line);border-radius:var(--r);padding:14px 16px}
.ev-top{display:flex;align-items:center;gap:9px;margin-bottom:5px}
.ev-type{font-family:"IBM Plex Mono",monospace;font-size:10.5px;font-weight:600;
 letter-spacing:.06em;padding:2px 7px;border-radius:4px}
.ev-m .ev-type{background:var(--accent-soft);color:var(--accent)}
.ev-l .ev-type{background:var(--mid-s);color:var(--mid)}
.ev-src{font-size:12px;color:var(--muted);font-family:"IBM Plex Mono",monospace}
.ev-unit{font-size:14.5px;font-weight:600;margin-bottom:6px}
.ev-pg{font-family:"IBM Plex Mono",monospace;font-size:11.5px;color:var(--muted);font-weight:400}
.ev-snip{font-size:13.5px;color:var(--ink-2);margin:0;line-height:1.6}

.gem{background:var(--surface);border:1px solid var(--line);border-radius:var(--r);padding:22px 26px}
.gem .gh{font-size:15.5px;font-weight:600;margin:20px 0 8px}
.gem .gh:first-child{margin-top:0}
.gem p{margin:0 0 10px;font-size:14.5px;color:var(--ink-2)}
.gem .gl{margin:0 0 12px;padding-left:19px}
.gem .gl li{font-size:14.5px;color:var(--ink-2);margin-bottom:5px}
.gem strong{color:var(--ink)}
.gem .quote{border-left:2px solid var(--line-2);padding-left:12px;color:var(--muted);font-size:13.5px}
hr.soft{border:none;border-top:1px solid var(--line);margin:16px 0}

.tablewrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{text-align:left;padding:8px 13px;border-bottom:1px solid var(--line)}
th{font-family:"IBM Plex Mono",monospace;font-size:11px;font-weight:600;letter-spacing:.07em;
 text-transform:uppercase;color:var(--muted);border-bottom:1px solid var(--line-2)}
.scroll-y{max-height:420px;overflow:auto;border:1px solid var(--line);border-radius:var(--r)}
.scroll-y th{position:sticky;top:0;background:var(--surface);z-index:1}

.call{border-left:3px solid var(--accent);background:var(--surface);
 border-radius:0 var(--r) var(--r) 0;padding:15px 19px;margin:20px 0}
.call.warn{border-left-color:var(--high)}
.call h4{margin:0 0 5px;font-size:15px;font-weight:600}
.call p{margin:0;font-size:14.5px;color:var(--ink-2)}
.call p+p{margin-top:7px}
ul.plain{margin:8px 0 0;padding-left:19px}
ul.plain li{font-size:14.5px;color:var(--ink-2);margin-bottom:5px}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
"""

# 로컬(file://)에서 열 때 한글이 깨지지 않으려면 doctype 과 charset 이 필요하다.
# 아티팩트로 게시할 때는 플랫폼이 <head> 를 감싸주므로 --embed 로 끈다.
STANDALONE = '--embed' not in sys.argv
HEAD = ('<!doctype html><html lang="ko"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">') if STANDALONE else ''
OPEN_BODY = '</head><body>' if STANDALONE else ''
TAIL = '</body></html>' if STANDALONE else ''

HTML = (
    HEAD +
    '<title>익산 작업 가이드 판독</title>\n'
    '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
    'family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans+KR:wght@400;500;600;700&display=swap">\n'
    '<style>' + CSS + '</style>\n'
    '<div class="wrap">\n'

    '<header>'
    '<p class="eyebrow">D 에이전트 · 실제 출력 캡처</p>'
    '<h1>익산 작업 가이드 판독</h1>'
    '<p class="lede">농가가 <strong>작업유형 하나를 고르면</strong> 화면에 실제로 무엇이 뜨는가. '
    '아래는 꾸민 예시가 아니라 <strong>Supabase·pgvector·Gemini를 모두 붙인 채 방금 실행한 결과</strong>다.</p>'
    '<div class="stamp"><span>작업유형 분뇨제거</span><span>농가 F001</span>'
    '<span>저장 경과 12일</span><span>Gemini ' + E(D['meta']['gemini_model']) + '</span></div>'
    '</header>\n'

    '<section style="margin-top:0">'
    '<h2><span class="n">01</span>데이터 출처 — 무엇을 보고 판단했는가</h2>'
    '<p class="sub">화면 상단에 항상 노출된다. 실데이터인지 고정데이터인지 사용자가 알 수 있어야 한다는 규약이다.</p>'
    '<div class="stats">' + srcpill('farm', '농가') + srcpill('risk_calendar', '위험도') +
    srcpill('storage', '저장일') + srcpill('rag', 'RAG 근거') + '</div></section>\n'

    '<section><h2><span class="n">02</span>추천 시간대 Top 3</h2>'
    '<p class="sub">연속 6시간 창의 위험도를 시간가중치로 합산해 <strong>낮은 순</strong>으로 뽑는다. '
    '코드가 확정하며 LLM이 관여하지 않는다.</p>' + windows(CASE['recommended']) +
    '<h3>회피 시간대 Top 3</h3>' + windows(CASE['avoid']) +
    '<div class="call"><h4>같은 시각인데 작업유형이 바뀌면 점수가 달라진다</h4>'
    '<p>액비살포는 작업 가중치가 1.50이라 분뇨제거(1.30)보다 같은 시각에도 점수가 높게 나온다.</p>'
    '<div class="tablewrap" style="margin-top:10px"><table><thead><tr>'
    '<th>등급</th><th>시작</th><th>6시간 위험</th><th>비교점수</th></tr></thead><tbody>' +
    alt_rows + '</tbody></table></div></div></section>\n'

    '<section><h2><span class="n">03</span>RAG 근거 — 왜 이 조치인가</h2>'
    '<p class="sub">Supabase <code>rag.chunks</code> 567청크에서 실무 3건·법령 3건을 각각 보장해 가져온다. '
    '파일명·단원·쪽수가 함께 나오므로 사람이 원문을 대조할 수 있다.</p>'
    '<div class="evs">' + evidence(CASE['evidence']) + '</div></section>\n'

    '<section><h2><span class="n">04</span>Gemini 설명 — 실제 응답 전문</h2>'
    '<p class="sub">위 <strong>02·03이 확정된 뒤에</strong> 호출된다. 프롬프트에 확정 결과와 RAG 근거가 '
    'JSON으로 들어가고, Gemini는 그것을 말로 풀어 쓸 뿐 숫자를 바꾸지 못한다.</p>'
    '<div class="gem">' + md2html(CASE['_gemini']) + '</div>'
    '<div class="call warn"><h4>키를 빼면 이 부분만 사라진다</h4>'
    '<p><code>GOOGLE_API_KEY</code>가 없으면 규칙 기반 문장으로 대체되고 '
    '<strong>추천·회피 시각과 점수는 그대로</strong> 나온다. 절대규칙 3의 실제 동작이다.</p></div></section>\n'

    '<section><h2><span class="n">05</span>가정과 한계 — 화면에 함께 뜬다</h2>'
    '<p class="sub">근거 없이 정한 값은 <code>[C]</code> 딱지를 달아 노출한다.</p>'
    '<div class="call"><h4>가정</h4><ul class="plain">' +
    ''.join('<li>' + E(a) + '</li>' for a in CASE['assumptions']) + '</ul></div>'
    '<div class="call warn"><h4>한계</h4><ul class="plain">' +
    ''.join('<li>' + E(l) + '</li>' for l in CASE['limitations']) + '</ul>'
    '<p style="margin-top:9px">법령 하드필터가 아직 없어 '
    '<strong>강우·결빙·부숙도·질소상한을 거치지 않은 결과</strong>다.</p></div></section>\n'

    '<section><h2><span class="n">06</span>7일 위험 캘린더 (원자료)</h2>'
    '<p class="sub">추천이 여기서 나온다. Supabase <code>risk_hourly</code> ' + str(n_cal) +
    '행 중 앞 30행. D+1~3은 1시간 단위, D+4~7은 일 단위'
    '(중기예보에 시간 해상도가 없어 시각 추천에서 제외).</p>'
    '<div class="scroll-y"><table><thead><tr><th>날짜</th><th>시각</th><th>위험지수</th>'
    '<th>등급</th><th>예측구간</th></tr></thead><tbody>' +
    cal_rows(D['calendar']['data']['items']) + '</tbody></table></div></section>\n'

    '<section><h2><span class="n">07</span>이 화면에 없는 것</h2>'
    '<div class="call warn"><h4>자유 질문 입력창</h4>'
    '<p>사용자는 작업유형 5개 중 선택만 한다. RAG 질의는 '
    '<code>&quot;{작업유형} 작업 전후 관리 기준&quot;</code>으로 자동 생성된다.</p></div>'
    '<div class="call warn"><h4>플룸 판정</h4>'
    '<p>플룸의 그룹선택은 <code>advisor/recommend.py</code>에만 있고 결과가 저장되지 않아, '
    'D는 두 그룹 중 높은 값을 택한다.</p></div>'
    '<div class="call warn"><h4>측정소 지도</h4>'
    '<p>익산악취24 원자료 미확보. fixture로 채우지 않고 <code>unavailable</code>을 그대로 표시한다.</p>'
    '</div></section>\n'
    '</div>\n')

if STANDALONE:
    # <head> 안에 charset 이 들어가야 file:// 로 열 때 한글이 안 깨진다.
    # <style> 까지가 head, 그 뒤가 body 다.
    cut = HTML.index('</style>') + len('</style>')
    HTML = HTML[:cut] + '</head><body>' + HTML[cut:] + '</body></html>'

io.open(DST, 'w', encoding='utf-8').write(HTML)
print('생성 완료:', DST, len(HTML), 'bytes')
