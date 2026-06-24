"""繁體中文顯示標籤(單一真相來源)。

後端洞察(narrator)與前端儀表板共用同一份對照表,避免兩邊各自翻譯而不一致。
資料庫裡的客群/分群名稱仍維持英文(資料完整性),僅在「呈現」時轉成中文。
"""

from __future__ import annotations

# RFM 規則式客群(11 類)
SEGMENT_ZH: dict[str, str] = {
    "Champions": "核心客戶",
    "Loyal Customers": "忠誠客戶",
    "Potential Loyalist": "潛在忠誠客戶",
    "New Customers": "新進客戶",
    "Promising": "看好客戶",
    "Need Attention": "需關注客戶",
    "About to Sleep": "即將沉睡",
    "At Risk": "流失風險客戶",
    "Can't Lose Them": "不能流失的客戶",
    "Hibernating": "沉睡客戶",
    "Lost": "已流失客戶",
}

# K-means 分群
CLUSTER_ZH: dict[str, str] = {
    "High-Value Active": "高價值・活躍",
    "High-Value Lapsing": "高價值・流失中",
    "Recent Low-Value": "近期・低價值",
    "Dormant Low-Value": "沉睡・低價值",
}

# 流失風險等級
RISK_ZH: dict[str, str] = {"low": "低", "medium": "中", "high": "高"}

# 各客群的建議行動(對應資料中的英文 Action)
ACTION_ZH: dict[str, str] = {
    "Reward loyalty; early access, referrals, VIP perks.": (
        "獎勵忠誠:搶先體驗、推薦獎勵、VIP 禮遇。"
    ),
    "Upsell higher-value lines; ask for reviews.": "推升至高價值商品線,並邀請留下評價。",
    "Membership / loyalty programme to deepen the habit.": "以會員/忠誠方案深化購買習慣。",
    "Strong onboarding; make the second purchase easy.": "強化新手引導,讓第二次購買更容易。",
    "Nurture with targeted offers to build frequency.": "以精準優惠培養,逐步提升回購頻率。",
    "Time-limited offers on recently browsed / bought lines.": (
        "對近期瀏覽/購買的商品提供限時優惠。"
    ),
    "Reactivation nudges before they churn.": "在流失前以喚醒訊息促使回訪。",
    "Win-back: personalised offers, remind them of value.": ("挽回:個人化優惠,提醒他們的價值。"),
    "High-touch win-back; they were valuable, don't lose them.": (
        "高接觸挽回:他們曾極具價值,務必留住。"
    ),
    "Low-cost reactivation; otherwise deprioritise spend.": ("低成本喚醒;否則降低投放優先度。"),
    "Minimal spend; only broad, cheap campaigns.": "投入最小化,僅用廣泛、低成本的活動。",
}


def segment_zh(name: str | None) -> str:
    return SEGMENT_ZH.get(name or "", name or "未知")


def risk_zh(level: str | None) -> str:
    return RISK_ZH.get(level or "", level or "—")


def action_zh(action: str | None) -> str:
    return ACTION_ZH.get(action or "", action or "")
