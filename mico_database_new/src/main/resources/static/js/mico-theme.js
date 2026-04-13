const micoTheme = {
    color: ['#0A84FF', '#30D158', '#FF9F0A', '#5E5CE6', '#FF453A', '#64D2FF'],
    backgroundColor: 'transparent',
    textStyle: {
        fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, "SF Pro Text", "PingFang SC", sans-serif'
    },
    title: {
        textStyle: {
            color: '#1d1d1f',
            fontWeight: 650
        }
    },
    tooltip: {
        backgroundColor: 'rgba(248,252,255,0.96)',
        borderColor: 'rgba(135,169,214,0.34)',
        borderWidth: 1,
        padding: [12, 16],
        textStyle: {
            color: '#1d1d1f',
            fontSize: 13
        },
        extraCssText: 'backdrop-filter:blur(16px) saturate(140%);-webkit-backdrop-filter:blur(16px) saturate(140%);box-shadow:0 14px 30px rgba(60,94,138,0.14);border-radius:16px;'
    },
    grid: {
        containLabel: true,
        left: '5%',
        right: '5%',
        bottom: '5%',
        top: '10%'
    },
    categoryAxis: {
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: { color: '#6e6e73', fontSize: 12 },
        splitLine: { 
            show: false,
            lineStyle: { type: 'dashed', color: 'rgba(161,181,209,0.25)' } 
        }
    },
    valueAxis: {
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: { color: '#6e6e73', fontSize: 12 },
        splitLine: { 
            show: true,
            lineStyle: { type: 'dashed', color: 'rgba(161,181,209,0.25)' } 
        }
    },
    line: {
        smooth: true,
        symbol: 'none',
        lineStyle: { width: 3 }
    },
    pie: {
        itemStyle: {
            borderWidth: 3,
            borderColor: '#fff'
        }
    }
};

if (typeof echarts !== 'undefined') {
    echarts.registerTheme('mico-medical', micoTheme);
}
