#!/usr/bin/env python3
"""Convert draw.io .drawio XML to Visio VDX format (directly openable in Visio).

Usage:
    python drawio2vdx.py input.drawio output.vdx
"""

import xml.etree.ElementTree as ET
import math
import sys
from pathlib import Path

PX_PER_IN = 96.0


def px2in(v):
    return v / PX_PER_IN


def esc(s):
    return (s.replace('&', '&amp;')
             .replace('<', '&lt;')
             .replace('>', '&gt;')
             .replace('"', '&quot;')
             .replace('\n', '&#10;'))


def clean_text(value):
    if not value:
        return ''
    text = value
    text = text.replace('<br>', '\n').replace('<br/>', '\n').replace('<br />', '\n')
    text = text.replace('&#10;', '\n')
    return text.strip()


class Drawio2Vdx:
    def __init__(self):
        self.shape_id = 1
        self.shapes_info = {}
        self.shape_xml = []

    def parse_style(self, style_str):
        d = {}
        if not style_str:
            return d
        for part in style_str.split(';'):
            part = part.strip()
            if not part:
                continue
            if '=' in part:
                k, v = part.split('=', 1)
                d[k.strip()] = v.strip()
            else:
                d[part] = True
        return d

    def y2v(self, y):
        return self.page_h_in - px2in(y)

    def convert(self, input_path, output_path):
        tree = ET.parse(input_path)
        root = tree.getroot()
        model = root.find('.//mxGraphModel')
        self.page_w_in = px2in(float(model.get('pageWidth', 1240)))
        self.page_h_in = px2in(float(model.get('pageHeight', 480)))

        cells = root.findall('.//mxCell')

        for cell in cells:
            if cell.get('id') in ('0', '1'):
                continue
            if cell.get('vertex') == '1':
                geom = cell.find('mxGeometry')
                if geom is not None:
                    xml = self.process_vertex(cell, geom)
                    if xml:
                        self.shape_xml.append(xml)

        for cell in cells:
            if cell.get('id') in ('0', '1'):
                continue
            if cell.get('edge') == '1':
                xml = self.process_edge(cell)
                if xml:
                    self.shape_xml.append(xml)

        vdx = self.build_vdx()
        Path(output_path).write_text(vdx, encoding='utf-8')
        print(f"VDX written to: {output_path}")
        print(f"Total shapes/connectors: {len(self.shape_xml)}")

    def process_vertex(self, cell, geom):
        cid = cell.get('id')
        style = self.parse_style(cell.get('style', ''))
        value = cell.get('value', '')
        x = float(geom.get('x', 0))
        y = float(geom.get('y', 0))
        w = float(geom.get('width', 0))
        h = float(geom.get('height', 0))
        rotation = float(style.get('rotation', 0))

        sid = self.shape_id
        self.shape_id += 1
        self.shapes_info[cid] = {'visio_id': sid, 'x': x, 'y': y, 'w': w, 'h': h}

        pin_x = px2in(x + w / 2)
        pin_y = self.y2v(y + h / 2)
        vw = px2in(w)
        vh = px2in(h)

        is_text = 'text' in style
        is_ellipse = 'ellipse' in style
        is_triangle = style.get('shape') == 'triangle'
        direction = style.get('direction', 'east')
        is_rounded = style.get('rounded') == '1'

        fill_color = style.get('fillColor', '#FFFFFF')
        no_fill = fill_color == 'none'
        stroke_color = style.get('strokeColor', '#000000')
        stroke_w = float(style.get('strokeWidth', 1.5))
        dashed = style.get('dashed') == '1'
        has_gradient = 'gradient' in style

        font_size = float(style.get('fontSize', 10))
        font_style_val = int(style.get('fontStyle', 0))
        bold = bool(font_style_val & 1)
        italic = bool(font_style_val & 2)

        align = style.get('align', 'center')
        valign = style.get('verticalAlign', 'middle')

        p = []
        p.append(f'<Shape ID="{sid}" Name="Shape.{sid}" Type="Shape">')

        # XForm
        p.append('<XForm>')
        p.append(f'<PinX Unit="IN">{pin_x:.5f}</PinX>')
        p.append(f'<PinY Unit="IN">{pin_y:.5f}</PinY>')
        p.append(f'<Width Unit="IN">{vw:.5f}</Width>')
        p.append(f'<Height Unit="IN">{vh:.5f}</Height>')
        p.append(f'<LocPinX Unit="IN" F="Width*0.5">{vw/2:.5f}</LocPinX>')
        p.append(f'<LocPinY Unit="IN" F="Height*0.5">{vh/2:.5f}</LocPinY>')
        if rotation:
            p.append(f'<Angle Unit="DEG">{-rotation}</Angle>')
        p.append('</XForm>')

        # Fill + Line
        if is_text:
            p.append('<Fill><FillPattern>0</FillPattern></Fill>')
            p.append('<Line><LinePattern>0</LinePattern></Line>')
        else:
            if no_fill:
                p.append('<Fill><FillPattern>0</FillPattern></Fill>')
            else:
                fc = '#808080' if has_gradient else fill_color
                p.append(f'<Fill><FillForegnd>{fc}</FillForegnd><FillPattern>1</FillPattern></Fill>')
            lp = [f'<LineColor>{stroke_color}</LineColor>',
                  f'<LineWeight Unit="PT">{stroke_w}</LineWeight>',
                  '<LinePattern>2</LinePattern>' if dashed else '<LinePattern>1</LinePattern>']
            if is_rounded:
                lp.append('<Rounding Unit="IN">0.04</Rounding>')
            p.append(f'<Line>{"".join(lp)}</Line>')

        # Char (text formatting)
        if value:
            cp = ['<Font>0</Font>', '<Color>#000000</Color>', f'<Size Unit="PT">{font_size}</Size>']
            sv = 0
            if bold: sv |= 1
            if italic: sv |= 2
            if sv:
                cp.append(f'<Style>{sv}</Style>')
            p.append(f'<Char IX="0">{"".join(cp)}</Char>')

        # Para (alignment)
        if value:
            ha = {'left': '0', 'center': '1', 'right': '2'}.get(align, '1')
            va = {'top': '0', 'middle': '1', 'bottom': '2'}.get(valign, '1')
            p.append(f'<Para IX="0"><HorzAlign>{ha}</HorzAlign></Para>')
            p.append(f'<TextBlock><VerticalAlign>{va}</VerticalAlign>'
                     '<TopMargin Unit="PT">2</TopMargin>'
                     '<BottomMargin Unit="PT">2</BottomMargin></TextBlock>')

        # Geometry
        if is_ellipse:
            p.append(self._ellipse_geom())
        elif is_triangle:
            p.append(self._triangle_geom(direction))
        elif not is_text:
            p.append(self._rect_geom())

        # Text
        if value:
            p.append(f'<Text>{esc(clean_text(value))}</Text>')

        p.append('</Shape>')
        return ''.join(p)

    def _rect_geom(self):
        return ('<Section N="Geometry" IX="0"><Cell N="NoFill" V="0"/>'
                '<Row T="MoveTo" IX="1"><Cell N="X" V="0" F="Width*0"/><Cell N="Y" V="0" F="Height*0"/></Row>'
                '<Row T="LineTo" IX="2"><Cell N="X" V="1" F="Width*1"/><Cell N="Y" V="0" F="Height*0"/></Row>'
                '<Row T="LineTo" IX="3"><Cell N="X" V="1" F="Width*1"/><Cell N="Y" V="1" F="Height*1"/></Row>'
                '<Row T="LineTo" IX="4"><Cell N="X" V="0" F="Width*0"/><Cell N="Y" V="1" F="Height*1"/></Row>'
                '<Row T="LineTo" IX="5"><Cell N="X" V="0" F="Width*0"/><Cell N="Y" V="0" F="Height*0"/></Row>'
                '</Section>')

    def _ellipse_geom(self):
        n = 36
        s = '<Section N="Geometry" IX="0"><Cell N="NoFill" V="0"/>'
        for i in range(n + 1):
            a = 2 * math.pi * i / n
            px = 0.5 + 0.5 * math.cos(a)
            py = 0.5 + 0.5 * math.sin(a)
            t = "MoveTo" if i == 0 else "LineTo"
            s += (f'<Row T="{t}" IX="{i+1}">'
                  f'<Cell N="X" V="{px:.4f}" F="Width*{px:.4f}"/>'
                  f'<Cell N="Y" V="{py:.4f}" F="Height*{py:.4f}"/></Row>')
        s += '</Section>'
        return s

    def _triangle_geom(self, direction):
        s = '<Section N="Geometry" IX="0"><Cell N="NoFill" V="0"/>'
        if direction == 'west':
            s += ('<Row T="MoveTo" IX="1"><Cell N="X" V="0" F="Width*0"/><Cell N="Y" V="0.5" F="Height*0.5"/></Row>'
                  '<Row T="LineTo" IX="2"><Cell N="X" V="1" F="Width*1"/><Cell N="Y" V="0" F="Height*0"/></Row>'
                  '<Row T="LineTo" IX="3"><Cell N="X" V="1" F="Width*1"/><Cell N="Y" V="1" F="Height*1"/></Row>'
                  '<Row T="LineTo" IX="4"><Cell N="X" V="0" F="Width*0"/><Cell N="Y" V="0.5" F="Height*0.5"/></Row>')
        else:
            s += ('<Row T="MoveTo" IX="1"><Cell N="X" V="0" F="Width*0"/><Cell N="Y" V="0" F="Height*0"/></Row>'
                  '<Row T="LineTo" IX="2"><Cell N="X" V="0" F="Width*0"/><Cell N="Y" V="1" F="Height*1"/></Row>'
                  '<Row T="LineTo" IX="3"><Cell N="X" V="1" F="Width*1"/><Cell N="Y" V="0.5" F="Height*0.5"/></Row>'
                  '<Row T="LineTo" IX="4"><Cell N="X" V="0" F="Width*0"/><Cell N="Y" V="0" F="Height*0"/></Row>')
        s += '</Section>'
        return s

    def process_edge(self, cell):
        style = self.parse_style(cell.get('style', ''))
        source_id = cell.get('source')
        target_id = cell.get('target')
        geom = cell.find('mxGeometry')

        # Begin point
        bx, by = None, None
        if source_id and source_id in self.shapes_info:
            si = self.shapes_info[source_id]
            ex = float(style.get('exitX', 0.5))
            ey = float(style.get('exitY', 0.5))
            bx = si['x'] + ex * si['w']
            by = si['y'] + ey * si['h']
        elif geom is not None:
            sp = geom.find('mxPoint[@as="sourcePoint"]')
            if sp is not None:
                bx = float(sp.get('x', 0))
                by = float(sp.get('y', 0))

        # End point
        ex_pt, ey_pt = None, None
        if target_id and target_id in self.shapes_info:
            ti = self.shapes_info[target_id]
            ax = float(style.get('entryX', 0.5))
            ay = float(style.get('entryY', 0.5))
            ex_pt = ti['x'] + ax * ti['w']
            ey_pt = ti['y'] + ay * ti['h']
        elif geom is not None:
            tp = geom.find('mxPoint[@as="targetPoint"]')
            if tp is not None:
                ex_pt = float(tp.get('x', 0))
                ey_pt = float(tp.get('y', 0))

        if bx is None or ex_pt is None:
            return None

        # Waypoints
        wps = []
        if geom is not None:
            arr = geom.find('Array[@as="points"]')
            if arr is not None:
                for pt in arr.findall('mxPoint'):
                    wps.append((float(pt.get('x', 0)), float(pt.get('y', 0))))

        all_pts = [(bx, by)] + wps + [(ex_pt, ey_pt)]
        visio_pts = [(px2in(px), self.y2v(py)) for px, py in all_pts]

        xs = [p[0] for p in visio_pts]
        ys = [p[1] for p in visio_pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        bw = max(max_x - min_x, 0.001)
        bh = max(max_y - min_y, 0.001)

        sid = self.shape_id
        self.shape_id += 1

        stroke_color = style.get('strokeColor', '#000000')
        stroke_w = float(style.get('strokeWidth', 1.5))
        dashed = style.get('dashed') == '1'
        end_arrow = style.get('endArrow', 'none')
        has_arrow = end_arrow in ('classic', 'open', 'block', 'diamond')

        p = []
        p.append(f'<Shape ID="{sid}" Name="Connector.{sid}" Type="Shape">')
        p.append('<XForm>')
        p.append(f'<PinX Unit="IN">{(min_x+max_x)/2:.5f}</PinX>')
        p.append(f'<PinY Unit="IN">{(min_y+max_y)/2:.5f}</PinY>')
        p.append(f'<Width Unit="IN">{bw:.5f}</Width>')
        p.append(f'<Height Unit="IN">{bh:.5f}</Height>')
        p.append(f'<LocPinX Unit="IN" F="Width*0.5">{bw/2:.5f}</LocPinX>')
        p.append(f'<LocPinY Unit="IN" F="Height*0.5">{bh/2:.5f}</LocPinY>')
        p.append('</XForm>')

        p.append('<Fill><FillPattern>0</FillPattern></Fill>')
        lp = [f'<LineColor>{stroke_color}</LineColor>',
              f'<LineWeight Unit="PT">{stroke_w}</LineWeight>',
              '<LinePattern>2</LinePattern>' if dashed else '<LinePattern>1</LinePattern>']
        if has_arrow:
            lp.append('<EndArrow>1</EndArrow>')
            lp.append('<BeginArrow>0</BeginArrow>')
        p.append(f'<Line>{"".join(lp)}</Line>')

        p.append('<Section N="Geometry" IX="0"><Cell N="NoFill" V="1"/>')
        for i, (vx, vy) in enumerate(visio_pts):
            rx = (vx - min_x) / bw if bw > 0 else 0
            ry = (vy - min_y) / bh if bh > 0 else 0
            t = "MoveTo" if i == 0 else "LineTo"
            p.append(f'<Row T="{t}" IX="{i+1}">'
                     f'<Cell N="X" V="{rx:.4f}" F="Width*{rx:.4f}"/>'
                     f'<Cell N="Y" V="{ry:.4f}" F="Height*{ry:.4f}"/></Row>')
        p.append('</Section>')

        p.append('</Shape>')
        return ''.join(p)

    def build_vdx(self):
        header = f'''<?xml version="1.0" encoding="utf-8"?>
<VisioDocument xmlns="http://schemas.microsoft.com/visio/2003/core" xmlns:rx="http://schemas.microsoft.com/officevisio/2006/extension">
<DocumentSettings><DefaultTemplate></DefaultTemplate><GlueSettings>9</GlueSettings></DocumentSettings>
<DocumentProperties>
<Creator>drawio2vdx</Creator>
<TimeCreated>2026-06-30T00:00:00Z</TimeCreated>
<TimeSaved>2026-06-30T00:00:00Z</TimeSaved>
</DocumentProperties>
<Pages>
<Page ID="0" Name="OFDM-FSO Link" NameU="OFDM-FSO Link">
<PageSheet>
<PageProps>
<PageWidth Unit="IN">{self.page_w_in:.4f}</PageWidth>
<PageHeight Unit="IN">{self.page_h_in:.4f}</PageHeight>
<DrawingScale Unit="IN">1</DrawingScale>
<DrawingScaleType>0</DrawingScaleType>
<PageLeftMargin Unit="IN">0.25</PageLeftMargin>
<PageRightMargin Unit="IN">0.25</PageRightMargin>
<PageTopMargin Unit="IN">0.25</PageTopMargin>
<PageBottomMargin Unit="IN">0.25</PageBottomMargin>
</PageProps>
</PageSheet>
<Shapes>
'''
        footer = '''</Shapes>
</Page>
</Pages>
</VisioDocument>'''
        return header + ''.join(self.shape_xml) + footer


if __name__ == '__main__':
    inp = sys.argv[1] if len(sys.argv) > 1 else 'ofdm_fso_link_visio.drawio'
    out = sys.argv[2] if len(sys.argv) > 2 else 'ofdm_fso_link_visio.vdx'
    Drawio2Vdx().convert(inp, out)
