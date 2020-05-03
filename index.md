---
layout: default
title: Home
nav_order: 1
permalink: /
---

## 소개

이 사이트는 음성 인식(speech recognition) 관련 이론들을 정리하기 위해 만들어 졌습니다. 머신러닝/딥러닝 일반에 관한 내용은 아래의 ratsgo's blog를, 최근 연구 동향이나 인사이트 관련은 ratsgo's insight notes을 참고하세요.

[ratsgo's blog](https://ratsgo.github.io){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 } [ratsgo's insight notes](https://ratsgo.github.io/insight-notes){: .btn .fs-5 .mb-4 .mb-md-0 }


---


## 홈페이지 구축에 사용한 오픈소스

이 페이지의 원본 소스는 다음의 저자가 만든 것을 활용한 것입니다. 진심으로 감사드립니다.

- [Just the Docs](http://patrickmarsceill.com) &copy; 2017-2019
- [utterances.es](https://utteranc.es/) &copy; 2020


---


## 라이센스

[CC BY-NC-SA 3.0 license](https://github.com/ratsgo/speechbook/blob/master/LICENSE)를 따릅니다. 다음 사항을 지키면 본 사이트에 있는 저작물들을 별도 허락 없이 자유롭게 사용할 수 있습니다.

- **저작권정보 표시** : 저작물 이용시 본 블로그 주소와 저작자를 표시해야 합니다
- **비영리** : 이 저작물은 영리 목적으로 이용할 수 없습니다.
- **동일조건 변경 허락** : 이 저작물을 변경(2차 저작물 작성 포함) 가능하나 자신이 만든 저작물에 본 저작물과 같은 이용조건(`CC BY-NC-SA 3.0`)을 적용해야 합니다.


---

## 토론하기

각 아티클 하단에 의견을 적을 수 있는 공간을 마련해 두었습니다. 내용 오류 및 보완, 오탈자 수정 등 자유롭게 의견 남겨 주시면 프로젝트 개선에 큰 도움이 될 것 같습니다. 

토론 이력은 모두 `github` 이슈(issue)와 연동됩니다. 일반적인 `github` 이슈와 동일하게 마크다운(markdown) 및 코드 스니펫(snippet) 양식이 모두 지원됩니다. 예컨대 코드1처럼 의견을 남기면 토론 코너에서 그림1과 같이 확인할 수 있습니다.

## **코드1** 토론 의견 남기기 예시
{: .no_toc .text-delta }
<div class="highlighter-rouge"><div class="highlight"><pre class="highlight"><code>**질문**
- 다음 코드를 실행하려면 어떤 환경이 필요한가요?
```python
print("hello") # 에러가 나는 부분은 이곳입니다.
```</code></pre></div></div>

## **그림1** 토론 코너에서 확인되는 화면
{: .no_toc .text-delta }
<img src="https://i.imgur.com/cogN6i1.png" width="600px" title="source: imgur.com" />

토론 코너에 의견을 남기고 나서 자신의 의견을 수정/삭제하고 싶을 수 있습니다. **이때는 `github` 이슈에 직접 들어가서 수정/삭제해야 합니다.** 그림2에서 파란색 사각형에 해당하는 링크를 누르면 해당 이슈로, 붉은색 링크를 따라가면 해당 코멘트로 바로 이동합니다.

## **그림2** 연결된 github issue로 이동하기
{: .no_toc .text-delta }
<img src="https://i.imgur.com/wjiC9ZB.png" width="600px" title="source: imgur.com" />

토론 코너에 사용한 오픈소스인 [utterances.es](https://utteranc.es)는 토론 의견만 노출할 뿐 커밋(commit) 로그, 이슈 연동 등은 트래킹하지 않습니다. 해당 게시물과 관련한 모든 개선 이력은 그림2의 파란색 링크의 이슈에 기재되어 있으니 참고하시면 좋을 것 같습니다.

---


## 기여
 
아래는 마스터(master) 브랜치에 풀리퀘스트(Pull Request)와 머지(merge)까지 진행해 이 프로젝트에 크게 기여해 주신 분들입니다. 토론과 개선 작업에 참여해 주셔서 진심으로 감사드립니다.

<ul class="list-style-none">
{% for contributor in site.github.contributors %}
  <li class="d-inline-block mr-1">
     <a href="{{ contributor.html_url }}"><img src="{{ contributor.avatar_url }}" width="32" height="32" alt="{{ contributor.login }}"/></a>
  </li>
{% endfor %}
</ul>

---

## 컨택포인트

제 이메일 주소 및 CV는 다음과 같습니다.

- ratsgo@naver.com
- [Curriculum Vitae](http://ratsgo.github.io/public/CV.pdf)

---