function initLightChart(domId, optionGenerator) {
    var dom = document.getElementById(domId);
    if (!dom) return;
    var myChart = echarts.init(dom, 'mico-medical', { renderer: 'canvas' });

    const baseOption = {
        textStyle: { fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, "SF Pro Text", "PingFang SC", sans-serif' },
        animationDuration: 1100,
        animationEasing: 'cubicOut'
    };

    const finalOption = { ...baseOption, ...optionGenerator() };
    myChart.setOption(finalOption);
    window.addEventListener("resize", () => myChart.resize());
}

function echarts_combined_stats() {
    initLightChart('echarts_combined_stats', () => {
        const stats = window.overviewStats || {};
        const total = stats.total_patients || '0';
        const disease = stats.most_common_disease_name || '--';
        const count = stats.most_common_disease_count || 0;

        return {
            grid: { top: 0, bottom: 0, left: 0, right: 0 },
            xAxis: { show: false },
            yAxis: { show: false },
            graphic: [
                {
                    type: 'group',
                    left: 'center',
                    top: '14%',
                    children: [
                        { type: 'text', style: { text: total, font: '700 44px -apple-system', fill: '#1d1d1f' } },
                        { type: 'text', top: 56, style: { text: '患者总记录数', font: '500 13px -apple-system', fill: '#6e6e73' } }
                    ]
                },
                {
                    type: 'line',
                    shape: { x1: 34, y1: 126, x2: 226, y2: 126 },
                    style: { stroke: '#e7e8ed', lineWidth: 1 }
                },
                {
                    type: 'group',
                    left: 'center',
                    top: '58%',
                    children: [
                        { type: 'text', style: { text: disease, font: '650 20px -apple-system', fill: '#0071e3' } },
                        { type: 'text', top: 32, style: { text: `高频疾病（${count}例）`, font: '500 12px -apple-system', fill: '#6e6e73' } }
                    ]
                }
            ]
        };
    });
}

function echarts_2() {
    initLightChart('echarts2', () => {
        let data = (window.diseaseDataFromFlask || [])
            .map(d => ({
                name: d.disease,
                value: d.count
            }))
            .sort((a, b) => b.value - a.value)
            .slice(0, 8)
            .reverse();

        const categories = data.map(item => item.name);
        const values = data.map(item => item.value);

        return {
            tooltip: {
                trigger: 'axis',
                axisPointer: { type: 'shadow', shadowStyle: { color: 'rgba(10, 132, 255, 0.08)' } },
                backgroundColor: 'rgba(248,252,255,0.96)',
                borderColor: 'rgba(135, 169, 214, 0.34)',
                borderWidth: 1,
                extraCssText: 'box-shadow: 0 14px 30px rgba(60, 94, 138, 0.14); border-radius: 16px;',
                textStyle: { color: '#1d1d1f' },
                padding: [10, 14],
                formatter: function(params) {
                    const point = params[0];
                    return point.name + '<br/>患者数量 <strong style="color:#0071e3;">' + Number(point.value || 0).toLocaleString('zh-CN') + '</strong>';
                }
            },
            grid: { left: '8%', right: '8%', bottom: '6%', top: '6%', containLabel: true },
            xAxis: {
                type: 'value',
                axisLine: { show: false },
                axisTick: { show: false },
                axisLabel: { color: '#6e6e73', fontSize: 12 },
                splitLine: { lineStyle: { type: 'dashed', color: 'rgba(161, 181, 209, 0.30)' } }
            },
            yAxis: {
                type: 'category',
                data: categories,
                axisLine: { show: false },
                axisTick: { show: false },
                axisLabel: { color: '#1d1d1f', fontSize: 12, fontWeight: 500 }
            },
            series: [{
                name: '患者数量',
                type: 'bar',
                data: values,
                barWidth: 18,
                itemStyle: {
                    borderRadius: [0, 11, 11, 0],
                    color: new echarts.graphic.LinearGradient(1, 0, 0, 0, [
                        { offset: 0, color: '#0A84FF' },
                        { offset: 1, color: '#7CC6FF' }
                    ]),
                    shadowBlur: 14,
                    shadowColor: 'rgba(10, 132, 255, 0.16)'
                },
                label: {
                    show: true,
                    position: 'right',
                    color: '#5d6b7b',
                    fontSize: 11,
                    formatter: function(params) {
                        return Number(params.value || 0).toLocaleString('zh-CN');
                    }
                }
            }]
        };
    });
}

function echarts_3() {
    initLightChart('echarts3', () => {
        let data = window.patientOriginData || [];
        if (!data.length) data = [];
        data = data
            .slice()
            .sort((a, b) => b.value - a.value)
            .slice(0, 6)
            .reverse();

        const cities = data.map(i => i.name);
        const values = data.map(i => i.value);

        return {
            grid: { left: '7%', right: '12%', bottom: '4%', top: '6%', containLabel: true },
            tooltip: {
                trigger: 'axis',
                axisPointer: { type: 'shadow', shadowStyle: { color: 'rgba(106, 168, 255, 0.08)' } },
                backgroundColor: 'rgba(248,252,255,0.96)',
                borderColor: 'rgba(135, 169, 214, 0.34)',
                borderWidth: 1,
                extraCssText: 'box-shadow: 0 14px 30px rgba(60, 94, 138, 0.14); border-radius: 16px;',
                padding: [10, 15]
            },
            xAxis: { show: false },
            yAxis: {
                type: 'category',
                data: cities,
                axisLine: { show: false },
                axisTick: { show: false },
                axisLabel: { color: '#1d1d1f', fontSize: 12, fontWeight: 500, margin: 12 }
            },
            series: [{
                type: 'bar',
                data: values,
                barWidth: 16,
                barCategoryGap: '54%',
                itemStyle: {
                    borderRadius: [0, 11, 11, 0],
                    shadowBlur: 14,
                    shadowColor: 'rgba(22, 163, 74, 0.16)',
                    color: function(params) {
                        const gradients = [
                            ['#0EA5E9', '#67E8F9'],
                            ['#0284C7', '#7DD3FC'],
                            ['#0891B2', '#5EEAD4'],
                            ['#2563EB', '#93C5FD'],
                            ['#0F766E', '#5EEAD4'],
                            ['#3B82F6', '#A5F3FC']
                        ];
                        const pair = gradients[params.dataIndex % gradients.length];
                        return new echarts.graphic.LinearGradient(1, 0, 0, 0, [
                            { offset: 0, color: pair[0] },
                            { offset: 1, color: pair[1] }
                        ]);
                    }
                },
                label: {
                    show: true,
                    position: 'right',
                    color: '#5d6b7b',
                    fontSize: 11,
                    formatter: function(params) {
                        return Number(params.value || 0).toLocaleString('en-US');
                    }
                }
            }]
        };
    });
}

function echarts_4() {
    initLightChart('echarts4', () => {
        let data = (window.diseaseDataFromFlask || [])
            .map(d => ({
                name: d.disease,
                value: d.count
            }))
            .sort((a, b) => b.value - a.value);

        if (data.length > 6) {
            const head = data.slice(0, 5);
            const tailTotal = data.slice(5).reduce((sum, item) => sum + item.value, 0);
            if (tailTotal > 0) {
                head.push({ name: '其他', value: tailTotal });
            }
            data = head;
        }

        return {
            tooltip: {
                trigger: 'item',
                backgroundColor: 'rgba(248,252,255,0.96)',
                borderColor: 'rgba(135, 169, 214, 0.34)',
                borderWidth: 1,
                extraCssText: 'box-shadow: 0 14px 30px rgba(60, 94, 138, 0.14); border-radius: 16px;'
            },
            legend: {
                orient: 'vertical',
                right: '4%',
                top: 'middle',
                icon: 'circle',
                itemWidth: 8,
                itemHeight: 8,
                itemGap: 12,
                textStyle: { color: '#6e6e73', fontSize: 11 }
            },
            series: [{
                name: '疾病构成',
                type: 'pie',
                radius: ['42%', '66%'],
                center: ['30%', '50%'],
                avoidLabelOverlap: false,
                itemStyle: {
                    borderRadius: 12,
                    borderColor: '#fff',
                    borderWidth: 3,
                    shadowBlur: 12,
                    shadowColor: 'rgba(72, 108, 160, 0.10)'
                },
                label: { show: false },
                labelLine: { show: false },
                data: data,
                color: ['#0A84FF', '#34C759', '#22C55E', '#38BDF8', '#6366F1', '#94A3B8']
            }],
            graphic: []
        };
    });
}

function echarts_5() {
    initLightChart('echarts5', () => {
        const data = window.ageGenderData || { age_brackets: [], male_counts: [], female_counts: [] };
        const crowded = (data.age_brackets || []).length > 8;

        return {
            tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
            legend: { right: 0, top: 0, icon: 'circle', textStyle: { color: '#6e6e73' } },
            grid: { left: '3%', right: '4%', bottom: crowded ? '20%' : '8%', top: '18%', containLabel: true },
            xAxis: {
                type: 'category',
                data: data.age_brackets,
                axisLine: { show: false },
                axisTick: { show: false },
                axisLabel: { color: '#6e6e73', interval: 0, rotate: crowded ? 28 : 0 }
            },
            yAxis: {
                type: 'value',
                splitLine: { lineStyle: { type: 'dashed', color: '#ececf0' } },
                axisLabel: { color: '#6e6e73' }
            },
            dataZoom: crowded ? [
                { type: 'inside', start: 0, end: 60 },
                { type: 'slider', height: 18, bottom: 8, start: 0, end: 60, borderColor: 'transparent' }
            ] : [],
            series: [
                { name: '男性', type: 'bar', stack: 'total', data: data.male_counts, itemStyle: { color: '#0071e3' }, barMaxWidth: 26 },
                { name: '女性', type: 'bar', stack: 'total', data: data.female_counts, itemStyle: { color: '#ff2d55', borderRadius: [4, 4, 0, 0] }, barMaxWidth: 26 }
            ]
        };
    });
}

$(document).ready(function() {
    $(".loading").fadeOut();
    echarts_combined_stats();
    echarts_2();
    echarts_3();
    echarts_4();
    echarts_5();
});
