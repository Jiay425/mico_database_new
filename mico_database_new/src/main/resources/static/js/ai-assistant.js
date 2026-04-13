(function () {
    function createDefaultConfig() {
        return {
            endpoint: "http://localhost:8088/api/ai/chat",
            title: "AI 助手",
            subtitle: "结合当前页面上下文，输出可执行结论",
            welcome: "我会基于当前页面数据给出简洁、可执行的分析结论。",
            starterPrompts: [],
            getContext: function () {
                return { page: "unknown", patientId: null, sampleId: null };
            }
        };
    }

    function safeJsonParse(text) {
        try {
            return JSON.parse(text);
        } catch (error) {
            return null;
        }
    }

    function sessionIdFor(page) {
        var key = "ai_assistant_session_id_" + page;
        var existing = sessionStorage.getItem(key);
        if (existing) return existing;
        var created = page + "-" + Date.now() + "-" + Math.random().toString(36).slice(2, 8);
        sessionStorage.setItem(key, created);
        return created;
    }

    function escapeHtml(text) {
        return String(text == null ? "" : text)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");
    }

    function normalizeText(text) {
        return String(text == null ? "" : text)
            .replace(/\s+/g, " ")
            .replace(/\bFo\.?\b/g, "")
            .replace(/\bnull\b/gi, "")
            .trim();
    }

    function shorten(text, maxLength) {
        var value = normalizeText(text);
        if (value.length <= maxLength) return value;
        return value.slice(0, maxLength) + "...";
    }

    function injectAssistantCardStyles() {
        var styleId = "ai-assistant-card-scroll-style";
        if (document.getElementById(styleId)) return;
        var style = document.createElement("style");
        style.id = styleId;
        style.textContent = ""
            + ".ai-card-content{white-space:pre-wrap;word-break:break-word;}"
            + ".ai-card-content.is-collapsed{max-height:110px;overflow-y:auto;padding-right:4px;}"
            + ".ai-card-content.is-expanded{max-height:220px;overflow-y:auto;}"
            + ".ai-card-controls{display:flex;justify-content:flex-end;gap:8px;margin-top:8px;}"
            + ".ai-card-btn{border:1px solid rgba(29,29,31,0.1);background:rgba(255,255,255,0.88);color:var(--ios-text-sec);border-radius:999px;padding:4px 10px;font-size:.75rem;font-weight:600;line-height:1.2;}"
            + ".ai-card-btn:hover{color:var(--ios-text);background:rgba(255,255,255,0.98);}";
        document.head.appendChild(style);
    }

    function toNumber(value, fallback) {
        var num = Number(value);
        return Number.isFinite(num) ? num : (fallback == null ? 0 : fallback);
    }

    function formatSigned(value, digits) {
        var num = toNumber(value, 0);
        var fixed = num.toFixed(digits == null ? 2 : digits);
        return (num >= 0 ? "+" : "") + fixed;
    }

    function isDifferenceQuestion(message) {
        var text = String(message || "").toLowerCase();
        if (!text) return true;
        return /差异|区别|偏离|健康组|健康|对照|一致|why|difference|deviation|consistent|consistency/.test(text);
    }

    function cleanTaxonName(raw) {
        var source = String(raw || "").trim();
        if (!source) return "";
        var parts = source.split(/[|;.\s]+/).filter(Boolean);
        for (var i = parts.length - 1; i >= 0; i -= 1) {
            var item = parts[i].replace(/^[a-z]__+/i, "").replace(/^_+|_+$/g, "");
            if (!item) continue;
            var low = item.toLowerCase();
            if (low === "unclassified" || low === "unknown") continue;
            return item.length > 20 ? (item.slice(0, 20) + "...") : item;
        }
        return source.length > 20 ? (source.slice(0, 20) + "...") : source;
    }

    function mapDeviationLevel(level) {
        var text = String(level || "").toLowerCase();
        if (text === "high") return "高偏离";
        if (text === "moderate") return "中等偏离";
        if (text === "mild") return "轻度偏离";
        if (text === "close_to_healthy") return "接近健康";
        return String(level || "待评估");
    }

    function fetchJsonOrNull(url) {
        return fetch(url).then(function (response) {
            if (!response.ok) return null;
            return response.text().then(function (text) {
                return safeJsonParse(text);
            });
        }).catch(function () {
            return null;
        });
    }

    function buildPatientDifferenceCard(topPayload, refPayload) {
        var topItems = (((topPayload || {}).data || {}).items || []);
        var topTaxa = topItems.slice(0, 5).map(function (item) {
            return cleanTaxonName(item && item.microbeName);
        }).filter(Boolean);

        var refData = (refPayload || {}).data || {};
        var comparisons = Array.isArray(refData.featureComparisons) ? refData.featureComparisons.slice() : [];
        comparisons.sort(function (a, b) {
            return toNumber(b && b.combinedScore, 0) - toNumber(a && a.combinedScore, 0);
        });
        var diffTop = comparisons.slice(0, 3).map(function (item) {
            var name = cleanTaxonName(item && item.microbeName);
            if (!name) return "";
            var log2fc = formatSigned(item && item.log2FoldChange, 2);
            var z = toNumber(item && item.zScore, 0).toFixed(2);
            return name + "(log2FC " + log2fc + ", Z=" + z + ")";
        }).filter(Boolean);

        var spatial = refData.spatialDeviation || {};
        var level = mapDeviationLevel(spatial.deviationLevel);
        var zScore = toNumber(spatial.distanceZScore, 0).toFixed(2);
        var percentile = toNumber(spatial.distancePercentile, 50).toFixed(1);

        if (!topTaxa.length && !diffTop.length && !spatial.deviationLevel) {
            return null;
        }

        var content = "Top丰度菌：" + (topTaxa.length ? topTaxa.join("、") : "暂无")
            + "；差异菌：" + (diffTop.length ? diffTop.join("、") : "暂无")
            + "；整体偏离：" + level + " (Z=" + zScore + "，百分位 " + percentile + "%)";

        return {
            type: "patient_difference",
            title: "健康差异量化",
            content: content
        };
    }

    function enrichResponseWithPatientDifference(response, context, message) {
        var safeResponse = response || {};
        if (!context || !context.patientId || !context.sampleId) {
            return Promise.resolve(safeResponse);
        }

        var patientId = encodeURIComponent(context.patientId);
        var sampleId = encodeURIComponent(context.sampleId);
        var topUrl = "/api/patients/" + patientId + "/samples/" + sampleId + "/top-features?limit=5";
        var refUrl = "/api/patients/" + patientId + "/samples/" + sampleId + "/healthy-reference?limit=5";

        return Promise.all([fetchJsonOrNull(topUrl), fetchJsonOrNull(refUrl)]).then(function (results) {
            var card = buildPatientDifferenceCard(results[0], results[1]);
            if (!card) return safeResponse;
            var mergedCards = Array.isArray(safeResponse.cards) ? safeResponse.cards.slice() : [];
            mergedCards = mergedCards.filter(function (item) {
                return !(item && item.type === "patient_difference");
            });
            mergedCards.unshift(card);
            safeResponse.cards = mergedCards;
            return safeResponse;
        });
    }

    function isNoiseCard(card) {
        var type = normalizeText(card && card.type).toLowerCase();
        return type === "multi_agent" || type === "agent_insights" || type === "hybrid_orchestration";
    }

    function normalizeCards(cards) {
        if (!Array.isArray(cards)) return [];
        var seen = {};
        var result = [];
        cards.forEach(function (card) {
            if (!card || isNoiseCard(card)) return;
            var type = normalizeText(card.type || "").toLowerCase();
            var title = normalizeText(card.title || "分析结果");
            var content = normalizeText(card.content || "");
            if (!title || !content) return;
            var dedupKey = type + "|" + title;
            if (seen[dedupKey]) return;
            seen[dedupKey] = true;
            result.push({
                type: type,
                title: title,
                content: content,
                longContent: content.length > 220
            });
        });
        result.sort(function (a, b) {
            var priority = {
                patient_difference: 0,
                healthy_reference: 1,
                prediction_link: 2
            };
            var pa = Object.prototype.hasOwnProperty.call(priority, a.type) ? priority[a.type] : 99;
            var pb = Object.prototype.hasOwnProperty.call(priority, b.type) ? priority[b.type] : 99;
            return pa - pb;
        });
        return result.slice(0, 5);
    }

    function normalizeActions(actions) {
        if (!Array.isArray(actions)) return [];
        var seen = {};
        var result = [];
        actions.forEach(function (item) {
            var action = normalizeText(item);
            if (!action) return;
            if (/agent/i.test(action)) return;
            if (seen[action]) return;
            seen[action] = true;
            result.push(action);
        });
        return result.slice(0, 3);
    }

    function renderCards(cards) {
        var normalized = normalizeCards(cards);
        if (!normalized.length) return "";
        return normalized.map(function (card, index) {
            var controls = "";
            if (card.longContent) {
                controls = "<div class=\"ai-card-controls\">"
                    + "<button type=\"button\" class=\"ai-card-btn ai-card-toggle\" data-expanded=\"false\">展开</button>"
                    + "<button type=\"button\" class=\"ai-card-btn ai-card-scroll\">下滑</button>"
                    + "</div>";
            }
            var contentClass = card.longContent ? "ai-card-content is-collapsed" : "ai-card-content";
            return "<div class=\"ai-card\" data-card-index=\"" + index + "\">"
                + "<div class=\"ai-card-title\">" + escapeHtml(card.title) + "</div>"
                + "<div class=\"" + contentClass + "\">" + escapeHtml(card.content) + "</div>"
                + controls
                + "</div>";
        }).join("");
    }

    function renderActions(actions) {
        var normalized = normalizeActions(actions);
        if (!normalized.length) return "";
        return "<div class=\"ai-actions\">"
            + normalized.map(function (action) {
                return "<button type=\"button\" class=\"ai-action-chip\" data-message=\"" + escapeHtml(action) + "\">" + escapeHtml(action) + "</button>";
            }).join("")
            + "</div>";
    }

    function buildAssistant(config) {
        var page = (config.getContext() || {}).page || "default";
        var sessionId = sessionIdFor(page);
        var root = document.createElement("div");
        root.className = "ai-assistant-shell";
        root.innerHTML = ""
            + "<button type=\"button\" class=\"ai-fab\" id=\"ai-fab\">"
            + "<i class=\"bi bi-stars\"></i><span>AI 助手</span>"
            + "</button>"
            + "<aside class=\"ai-panel\" id=\"ai-panel\" aria-hidden=\"true\">"
            + "<div class=\"ai-panel-header\">"
            + "<div>"
            + "<span class=\"section-kicker\">智能分析</span>"
            + "<h3>" + escapeHtml(config.title) + "</h3>"
            + "<p>" + escapeHtml(config.subtitle) + "</p>"
            + "</div>"
            + "<button type=\"button\" class=\"ai-panel-close\" id=\"ai-close\"><i class=\"bi bi-x-lg\"></i></button>"
            + "</div>"
            + "<div class=\"ai-panel-body\">"
            + "<div class=\"ai-thread\" id=\"ai-thread\">"
            + "<div class=\"ai-message ai-message-assistant\">"
            + "<div class=\"ai-bubble\">"
            + "<strong>AI 助手</strong>"
            + "<p>" + escapeHtml(config.welcome) + "</p>"
            + "</div>"
            + "</div>"
            + "</div>"
            + "<div class=\"ai-starters\" id=\"ai-starters\"></div>"
            + "<form class=\"ai-input-bar\" id=\"ai-form\">"
            + "<textarea id=\"ai-input\" rows=\"3\" placeholder=\"例如：当前样本和健康组相比有什么区别？\"></textarea>"
            + "<button type=\"submit\" id=\"ai-submit\"><i class=\"bi bi-send-fill\"></i></button>"
            + "</form>"
            + "</div>"
            + "</aside>";
        document.body.appendChild(root);

        var panel = root.querySelector("#ai-panel");
        var fab = root.querySelector("#ai-fab");
        var closeBtn = root.querySelector("#ai-close");
        var form = root.querySelector("#ai-form");
        var input = root.querySelector("#ai-input");
        var thread = root.querySelector("#ai-thread");
        var starters = root.querySelector("#ai-starters");
        var submit = root.querySelector("#ai-submit");

        function openPanel(prefill) {
            panel.classList.add("open");
            panel.setAttribute("aria-hidden", "false");
            if (prefill) {
                input.value = prefill;
            }
            window.setTimeout(function () { input.focus(); }, 50);
        }

        function closePanel() {
            panel.classList.remove("open");
            panel.setAttribute("aria-hidden", "true");
        }

        function appendUserMessage(message) {
            var node = document.createElement("div");
            node.className = "ai-message ai-message-user";
            node.innerHTML = "<div class=\"ai-bubble\"><p>" + escapeHtml(message) + "</p></div>";
            thread.appendChild(node);
            thread.scrollTop = thread.scrollHeight;
        }

        function appendAssistantMessage(response) {
            var summary = normalizeText(response.summary || "");
            if (!summary) {
                summary = "已完成分析，请继续追问“差异在哪里”或“下一步怎么做”。";
            }
            summary = shorten(summary, 180);

            var cardsHtml = renderCards(response.cards);
            var actionsHtml = renderActions(response.actions);

            var node = document.createElement("div");
            node.className = "ai-message ai-message-assistant";
            node.innerHTML = "<div class=\"ai-bubble\">"
                + "<strong>AI 助手</strong>"
                + "<p>" + escapeHtml(summary) + "</p>"
                + cardsHtml
                + actionsHtml
                + "</div>";
            thread.appendChild(node);
            thread.scrollTop = thread.scrollHeight;
        }

        function appendErrorMessage(message) {
            var node = document.createElement("div");
            node.className = "ai-message ai-message-assistant";
            node.innerHTML = "<div class=\"ai-bubble ai-bubble-warning\"><strong>调用失败</strong><p>" + escapeHtml(message) + "</p></div>";
            thread.appendChild(node);
            thread.scrollTop = thread.scrollHeight;
        }

        function setLoading(loading) {
            submit.disabled = loading;
            submit.classList.toggle("is-loading", loading);
            if (loading) {
                submit.innerHTML = "<span class=\"spinner-border spinner-border-sm\"></span>";
            } else {
                submit.innerHTML = "<i class=\"bi bi-send-fill\"></i>";
            }
        }

        function sendMessage(message) {
            if (!message || !message.trim()) return;
            var cleanMessage = message.trim();
            var payload = {
                sessionId: sessionId,
                message: cleanMessage,
                context: config.getContext ? config.getContext() : { page: "unknown", patientId: null, sampleId: null }
            };

            appendUserMessage(cleanMessage);
            input.value = "";
            setLoading(true);

            fetch(config.endpoint, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            })
                .then(function (response) {
                    return response.text().then(function (text) {
                        var data = safeJsonParse(text);
                        if (!response.ok) {
                            throw new Error(data && data.error ? data.error : ("HTTP " + response.status));
                        }
                        return data || {};
                    });
                })
                .then(function (data) {
                    return enrichResponseWithPatientDifference(data, payload.context, cleanMessage);
                })
                .then(function (enhanced) {
                    appendAssistantMessage(enhanced || {});
                })
                .catch(function (error) {
                    appendErrorMessage(error.message || "AI 服务暂时不可用，请确认 8088 已启动。");
                })
                .finally(function () {
                    setLoading(false);
                });
        }

        fab.addEventListener("click", function () { openPanel(); });
        closeBtn.addEventListener("click", closePanel);

        form.addEventListener("submit", function (event) {
            event.preventDefault();
            sendMessage(input.value);
        });

        root.addEventListener("click", function (event) {
            var target = event.target;
            var chip = target.closest(".ai-action-chip");
            if (chip) {
                sendMessage(chip.getAttribute("data-message") || chip.textContent || "");
                return;
            }
            var toggleBtn = target.closest(".ai-card-toggle");
            if (toggleBtn) {
                var cardNode = toggleBtn.closest(".ai-card");
                if (!cardNode) return;
                var contentNode = cardNode.querySelector(".ai-card-content");
                if (!contentNode) return;
                var expanded = toggleBtn.getAttribute("data-expanded") === "true";
                if (expanded) {
                    contentNode.classList.remove("is-expanded");
                    toggleBtn.setAttribute("data-expanded", "false");
                    toggleBtn.textContent = "展开";
                } else {
                    contentNode.classList.add("is-expanded");
                    toggleBtn.setAttribute("data-expanded", "true");
                    toggleBtn.textContent = "收起";
                }
                return;
            }
            var scrollBtn = target.closest(".ai-card-scroll");
            if (scrollBtn) {
                var wrap = scrollBtn.closest(".ai-card");
                if (!wrap) return;
                var content = wrap.querySelector(".ai-card-content");
                if (!content) return;
                if (!content.classList.contains("is-expanded")) {
                    content.classList.add("is-expanded");
                }
                content.scrollBy({ top: 120, left: 0, behavior: "smooth" });
                var pairedToggle = wrap.querySelector(".ai-card-toggle");
                if (pairedToggle) {
                    pairedToggle.setAttribute("data-expanded", "true");
                    pairedToggle.textContent = "收起";
                }
                return;
            }
            var externalOpen = target.closest(".ai-assistant-open");
            if (externalOpen) {
                var preset = externalOpen.getAttribute("data-ai-message") || "";
                openPanel(preset);
            }
        });

        (config.starterPrompts || []).forEach(function (prompt) {
            var btn = document.createElement("button");
            btn.type = "button";
            btn.className = "ai-starter-chip";
            btn.setAttribute("data-message", prompt);
            btn.textContent = prompt;
            starters.appendChild(btn);
        });

        starters.addEventListener("click", function (event) {
            var btn = event.target.closest(".ai-starter-chip");
            if (!btn) return;
            openPanel(btn.getAttribute("data-message"));
        });
    }

    document.addEventListener("DOMContentLoaded", function () {
        injectAssistantCardStyles();
        var config = Object.assign(createDefaultConfig(), window.aiAssistantConfig || {});
        buildAssistant(config);
    });
})();
