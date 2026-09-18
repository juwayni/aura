"""
Aura v4 — Production-grade ARC + Arenas + CSP compiler targeting C11.
Features: Structs, Methods, Enums, Maps, Slices, Channels, Defer, M:N Scheduler.
"""
from __future__ import annotations
import sys, os, subprocess, tempfile, shutil
from typing import Optional, List, Dict, Tuple, Any

# ═══════════════════════════════════════════════════════════════════════════
# 1 — LEXER
# ═══════════════════════════════════════════════════════════════════════════

KEYWORDS = {
    'let','const','func','return','if','else','for','spawn','chan',
    'arena','extern','true','false','nil','import','type','struct',
    'break','continue','make','as','in','result','defer','enum','map'
}

MULTI_OPS = [
    '..=','...','..','<-','==','!=','<=','>=','&&','||',':=','->',
    '+=','-=','*=','/=','%=','++','--'
]


class Token:
    __slots__ = ('kind', 'value', 'line', 'col')
    def __init__(self, k: str, v: Any, ln: int, c: int):
        self.kind, self.value, self.line, self.col = k, v, ln, c
    def __repr__(self):
        return f"Token({self.kind},{self.value!r}@{self.line}:{self.col})"


class AuraError(Exception):
    def __init__(self, msg: str, line: int = 0, col: int = 0):
        super().__init__(msg)
        self.line, self.col = line, col


class Lexer:
    def __init__(self, src: str):
        self.src, self.pos, self.line, self.col = src, 0, 1, 1

    def _err(self, m: str):
        raise AuraError(f"[Lexer] {self.line}:{self.col}: {m}", self.line, self.col)

    def _peek(self, o: int = 0) -> str:
        p = self.pos + o
        return self.src[p] if p < len(self.src) else ''

    def _adv(self) -> str:
        c = self.src[self.pos]; self.pos += 1
        if c == '\n': self.line += 1; self.col = 1
        else: self.col += 1
        return c

    def _skip_ws(self):
        while self.pos < len(self.src):
            c = self._peek()
            if c in ' \t\r\n': self._adv()
            elif c == '/' and self._peek(1) == '/':
                while self.pos < len(self.src) and self._peek() != '\n': self._adv()
            elif c == '/' and self._peek(1) == '*':
                self._adv(); self._adv()
                while self.pos < len(self.src) and not (self._peek() == '*' and self._peek(1) == '/'):
                    self._adv()
                if self.pos >= len(self.src): self._err("unterminated block comment")
                self._adv(); self._adv()
            else: break

    def _ident(self) -> str:
        s = ''
        while self.pos < len(self.src) and (self._peek().isalnum() or self._peek() == '_'):
            s += self._adv()
        return s

    def _num(self) -> str:
        s, dot = '', False
        while self.pos < len(self.src):
            c = self._peek()
            if c.isdigit(): s += self._adv()
            elif c == '_': self._adv()
            elif c == '.' and not dot and self._peek(1).isdigit():
                dot = True; s += self._adv()
            else: break
        return s

    def _str(self) -> str:
        q = self._adv(); s = ''
        while self.pos < len(self.src) and self._peek() != q:
            if self._peek() == '\\':
                self._adv()
                if self.pos >= len(self.src): self._err("unterminated escape")
                c = self._adv()
                s += {'n':'\n','t':'\t','r':'\r','\\':'\\','"':'"',"'":"'",'0':'\0'}.get(c, c)
            else: s += self._adv()
        if self.pos >= len(self.src): self._err("unterminated string")
        self._adv(); return s

    def tokenize(self) -> List[Token]:
        toks: List[Token] = []
        while True:
            self._skip_ws()
            if self.pos >= len(self.src): break
            ln, cl, c = self.line, self.col, self._peek()
            if c.isalpha() or c == '_':
                i = self._ident()
                toks.append(Token(i if i in KEYWORDS else 'IDENT', i, ln, cl))
            elif c.isdigit():
                n = self._num()
                toks.append(Token('FLOAT' if '.' in n else 'NUMBER', n, ln, cl))
            elif c in '"\'':
                toks.append(Token('STRING', self._str(), ln, cl))
            else:
                for op in MULTI_OPS:
                    if self.src[self.pos:self.pos+len(op)] == op:
                        for _ in op: self._adv()
                        toks.append(Token(op, op, ln, cl)); break
                else:
                    self._adv(); toks.append(Token(c, c, ln, cl))
        toks.append(Token('EOF', None, self.line, self.col))
        return toks


# ═══════════════════════════════════════════════════════════════════════════
# 2 — AST
# ═══════════════════════════════════════════════════════════════════════════

class Node: line: int = 0

class TypeRef(Node):
    def __init__(self, name: str, args: Optional[List[TypeRef]] = None, line: int = 0):
        self.name, self.args, self.line = name, args or [], line
    def __repr__(self):
        return self.name if not self.args else f"{self.name}<{','.join(map(repr,self.args))}>"

class Program(Node):
    def __init__(self, decls: List[Node]): self.decls = decls

class StructDecl(Node):
    def __init__(self, name: str, fields: List[Tuple[str, TypeRef]], line: int = 0):
        self.name, self.fields, self.line = name, fields, line

class EnumDecl(Node):
    def __init__(self, name: str, variants: List[Tuple[str, List[TypeRef]]], line: int = 0):
        self.name, self.variants, self.line = name, variants, line

class ConstDecl(Node):
    def __init__(self, name: str, init: Node, line: int = 0):
        self.name, self.init, self.line = name, init, line

class ExternBlock(Node):
    def __init__(self, abi: str, decls: List[FuncDecl], line: int = 0):
        self.abi, self.decls, self.line = abi, decls, line

class Param:
    def __init__(self, name: str, type_: Optional[TypeRef], variadic: bool = False):
        self.name, self.type, self.variadic = name, type_, variadic

class Receiver:
    def __init__(self, name: str, type_: TypeRef): self.name, self.type = name, type_

class FuncDecl(Node):
    def __init__(self, name: str, params: List[Param], ret_type: Optional[TypeRef],
                 body: Optional[Block], is_extern: bool = False, abi: Optional[str] = None,
                 receiver: Optional[Receiver] = None, line: int = 0):
        self.name, self.params, self.ret_type = name, params, ret_type
        self.body, self.is_extern, self.abi = body, is_extern, abi
        self.receiver, self.line = receiver, line

class VarDecl(Node):
    def __init__(self, name: str, type_: Optional[TypeRef], init: Optional[Node], line: int = 0):
        self.name, self.type, self.init, self.line = name, type_, init, line

class ReturnStmt(Node):
    def __init__(self, value: Optional[Node], line: int = 0): self.value, self.line = value, line

class DeferStmt(Node):
    def __init__(self, stmt: Node, line: int = 0): self.stmt, self.line = stmt, line

class IfStmt(Node):
    def __init__(self, cond: Node, then: Block, els: Optional[Node], line: int = 0):
        self.cond, self.then, self.els, self.line = cond, then, els, line

class ForStmt(Node):
    def __init__(self, init: Optional[Node], cond: Optional[Node], post: Optional[Node],
                 body: Block, line: int = 0):
        self.init, self.cond, self.post, self.body, self.line = init, cond, post, body, line

class ForInStmt(Node):
    def __init__(self, var: str, iterable: Node, body: Block, line: int = 0):
        self.var, self.iterable, self.body, self.line = var, iterable, body, line
        self.kind: Optional[str] = None
        self.elem_type: Optional[Tuple] = None
        self.inclusive: bool = False

class RangeExpr(Node):
    def __init__(self, lo: Node, hi: Node, inclusive: bool, line: int = 0):
        self.lo, self.hi, self.inclusive, self.line = lo, hi, inclusive, line

class Block(Node):
    def __init__(self, stmts: List[Node], line: int = 0): self.stmts, self.line = stmts, line

class ExprStmt(Node):
    def __init__(self, expr: Node, line: int = 0): self.expr, self.line = expr, line

class SpawnStmt(Node):
    def __init__(self, call: CallExpr, line: int = 0): self.call, self.line = call, line

class ArenaStmt(Node):
    def __init__(self, body: Block, line: int = 0): self.body, self.line = body, line

class BreakStmt(Node): pass
class ContinueStmt(Node): pass

class AssignExpr(Node):
    def __init__(self, target: Node, value: Node, line: int = 0, op: str = '='):
        self.target, self.value, self.line, self.op = target, value, line, op

class BinaryExpr(Node):
    def __init__(self, op: str, l: Node, r: Node, line: int = 0):
        self.op, self.left, self.right, self.line = op, l, r, line

class UnaryExpr(Node):
    def __init__(self, op: str, operand: Node, line: int = 0):
        self.op, self.operand, self.line = op, operand, line

class CallExpr(Node):
    def __init__(self, func: Node, args: List[Node], line: int = 0):
        self.func, self.args, self.line = func, args, line

class Ident(Node):
    def __init__(self, name: str, line: int = 0): self.name, self.line = name, line

class IntLit(Node):
    def __init__(self, v: int, line: int = 0): self.value, self.line = v, line

class FloatLit(Node):
    def __init__(self, v: float, line: int = 0): self.value, self.line = v, line

class BoolLit(Node):
    def __init__(self, v: bool, line: int = 0): self.value, self.line = v, line

class StringLit(Node):
    def __init__(self, v: str, line: int = 0): self.value, self.line = v, line

class NilLit(Node):
    def __init__(self, line: int = 0): self.line = line

class ChanSend(Node):
    def __init__(self, chan: Node, value: Node, line: int = 0):
        self.chan, self.value, self.line = chan, value, line

class ChanRecv(Node):
    def __init__(self, chan: Node, line: int = 0): self.chan, self.line = chan, line

class ChanMake(Node):
    def __init__(self, elem: TypeRef, cap: Node, line: int = 0):
        self.elem_type, self.capacity, self.line = elem, cap, line

class MapMake(Node):
    def __init__(self, key_type: TypeRef, val_type: TypeRef, line: int = 0):
        self.key_type, self.val_type, self.line = key_type, val_type, line

class MemberAccess(Node):
    def __init__(self, obj: Node, field: str, line: int = 0):
        self.obj, self.field, self.line = obj, field, line

class IndexExpr(Node):
    def __init__(self, obj: Node, index: Node, line: int = 0):
        self.obj, self.index, self.line = obj, index, line

class SliceLit(Node):
    def __init__(self, elems: List[Node], line: int = 0):
        self.elements, self.line = elems, line

class StructLit(Node):
    def __init__(self, name: str, inits: List[Tuple[str, Node]], line: int = 0):
        self.name, self.inits, self.line = name, inits, line

class TryExpr(Node):
    def __init__(self, expr: Node, line: int = 0): self.expr, self.line = expr, line


# ═══════════════════════════════════════════════════════════════════════════
# 3 — PARSER
# ═══════════════════════════════════════════════════════════════════════════

PRIMITIVES = {'int','float','bool','string','void'}
COMPOUND_OPS = {'+=','-=','*=','/=','%='}


class Parser:
    def __init__(self, toks: List[Token]): self.toks, self.i = toks, 0
    def _peek(self, o: int = 0) -> Token: return self.toks[min(self.i+o, len(self.toks)-1)]
    def _at(self, k: str, v: Any = None) -> bool:
        t = self._peek(); return t.kind == k and (v is None or t.value == v)
    def _eat(self, k: str, v: Any = None) -> Token:
        t = self._peek()
        if not self._at(k, v):
            exp = v if v else k
            raise AuraError(f"[Parser] {t.line}:{t.col}: expected '{exp}', got '{t.value or t.kind}'", t.line, t.col)
        self.i += 1; return t
    def _maybe(self, k: str, v: Any = None) -> Optional[Token]:
        if self._at(k, v): return self._eat(k, v)
        return None

    # ---- types ----
    def parse_type(self) -> TypeRef:
        t = self._peek()
        if t.kind == '*':
            self._eat('*')
            return TypeRef('ptr', [self.parse_type()], t.line)
        if t.kind == 'chan':
            self._eat('chan')
            return TypeRef('chan', [self.parse_type()], t.line)
        if t.kind == '[':
            self._eat('['); self._eat(']')
            return TypeRef('slice', [self.parse_type()], t.line)
        if t.kind == 'map':
            self._eat('map'); self._eat('[')
            kt = self.parse_type(); self._eat(']')
            vt = self.parse_type()
            return TypeRef('map', [kt, vt], t.line)
        if t.kind == 'result':
            self._eat('result'); self._eat('[')
            ok = self.parse_type(); self._eat(',')
            er = self.parse_type(); self._eat(']')
            return TypeRef('result', [ok, er], t.line)
        if t.kind == 'IDENT' or t.kind in PRIMITIVES:
            self._eat(t.kind)
            name, args = t.value, []
            if self._at('<'):
                self._eat('<'); args.append(self.parse_type())
                while self._maybe(','): args.append(self.parse_type())
                self._eat('>')
            return TypeRef(name, args, t.line)
        raise AuraError(f"[Parser] {t.line}:{t.col}: expected type, got '{t.value or t.kind}'", t.line, t.col)

    # ---- top-level ----
    def parse_program(self) -> Program:
        decls: List[Node] = []
        while not self._at('EOF'):
            if self._at('extern'):  decls.append(self.parse_extern())
            elif self._at('struct'): decls.append(self.parse_struct())
            elif self._at('enum'):   decls.append(self.parse_enum())
            elif self._at('const'):  decls.append(self.parse_const())
            elif self._at('func'):   decls.append(self.parse_func())
            else:
                t = self._peek()
                raise AuraError(f"[Parser] {t.line}:{t.col}: unexpected declaration '{t.value or t.kind}'", t.line, t.col)
        return Program(decls)

    def parse_struct(self) -> StructDecl:
        ln = self._peek().line
        self._eat('struct')
        name = self._eat('IDENT').value
        self._eat('{')
        fields = []
        while not self._at('}'):
            fn = self._eat('IDENT').value
            ft = self.parse_type()
            self._maybe(';')
            fields.append((fn, ft))
        self._eat('}')
        return StructDecl(name, fields, ln)

    def parse_enum(self) -> EnumDecl:
        ln = self._peek().line
        self._eat('enum')
        name = self._eat('IDENT').value
        self._eat('{')
        variants = []
        while not self._at('}'):
            vn = self._eat('IDENT').value
            types = []
            if self._at('('):
                self._eat('(')
                if not self._at(')'):
                    types.append(self.parse_type())
                    while self._maybe(','): types.append(self.parse_type())
                self._eat(')')
            self._maybe(',')
            variants.append((vn, types))
        self._eat('}')
        return EnumDecl(name, variants, ln)

    def parse_const(self) -> ConstDecl:
        ln = self._peek().line
        self._eat('const')
        name = self._eat('IDENT').value
        self._maybe('=') or self._eat('=')
        init = self.parse_expr()
        self._maybe(';')
        return ConstDecl(name, init, ln)

    def parse_extern(self) -> ExternBlock:
        ln = self._peek().line
        self._eat('extern')
        abi = self._eat('STRING').value
        self._eat('{')
        decls = []
        while not self._at('}'): decls.append(self.parse_func(is_extern=True, abi=abi))
        self._eat('}')
        return ExternBlock(abi, decls, ln)

    def parse_func(self, is_extern: bool = False, abi: Optional[str] = None) -> FuncDecl:
        ln = self._peek().line
        self._eat('func')
        receiver = None
        if self._at('('):
            self._eat('(')
            rn = self._eat('IDENT').value
            if self._at('*'):
                self._eat('*')
                rt = TypeRef('ptr', [self.parse_type()], ln)
            else:
                rt = self.parse_type()
            self._eat(')')
            receiver = Receiver(rn, rt)
        name = self._eat('IDENT').value
        self._eat('(')
        params: List[Param] = []
        if not self._at(')'):
            while True:
                if self._at('...'):
                    self._eat('...'); params.append(Param('', None, True)); break
                pn = self._eat('IDENT').value
                pt = self.parse_type()
                params.append(Param(pn, pt))
                if not self._maybe(','): break
        self._eat(')')
        ret = None if self._at('{') or (is_extern and (self._at(';') or self._at('}'))) else self.parse_type()
        body = None
        if is_extern: self._maybe(';')
        else: body = self.parse_block()
        return FuncDecl(name, params, ret, body, is_extern, abi, receiver, ln)

    # ---- statements ----
    def parse_block(self) -> Block:
        ln = self._peek().line
        self._eat('{')
        stmts = []
        while not self._at('}'): stmts.append(self.parse_stmt())
        self._eat('}')
        return Block(stmts, ln)

    def parse_stmt(self) -> Node:
        t = self._peek()
        if t.kind == 'let':     return self.parse_var_decl()
        if t.kind == 'defer':   return self.parse_defer()
        if t.kind == 'return':  return self.parse_return()
        if t.kind == 'if':      return self.parse_if()
        if t.kind == 'for':     return self.parse_for()
        if t.kind == 'spawn':   return self.parse_spawn()
        if t.kind == 'arena':   return self.parse_arena()
        if t.kind == '{':       return self.parse_block()
        if t.kind == 'break':   self._eat('break'); self._maybe(';'); return BreakStmt()
        if t.kind == 'continue':self._eat('continue'); self._maybe(';'); return ContinueStmt()
        e = self.parse_expr()
        self._maybe(';')
        return ExprStmt(e, t.line)

    def parse_var_decl(self) -> VarDecl:
        ln = self._peek().line
        self._eat('let')
        name = self._eat('IDENT').value
        t = None
        if not self._at('=') and not self._at(';'):
            t = self.parse_type()
        init = None
        if self._maybe('='):
            init = self.parse_expr()
        self._maybe(';')
        return VarDecl(name, t, init, ln)

    def parse_defer(self) -> DeferStmt:
        ln = self._peek().line
        self._eat('defer')
        stmt = self.parse_stmt()
        return DeferStmt(stmt, ln)

    def parse_return(self) -> ReturnStmt:
        ln = self._peek().line
        self._eat('return')
        val = None
        if not self._at('}') and not self._at(';'): val = self.parse_expr()
        self._maybe(';')
        return ReturnStmt(val, ln)

    def parse_if(self) -> IfStmt:
        ln = self._peek().line
        self._eat('if')
        cond = self.parse_expr()
        then = self.parse_block()
        els = None
        if self._maybe('else'):
            els = self.parse_if() if self._at('if') else self.parse_block()
        return IfStmt(cond, then, els, ln)

    def parse_for(self) -> Node:
        ln = self._peek().line
        self._eat('for')
        if self._at('{'): return ForStmt(None, None, None, self.parse_block(), ln)
        # for x in iter
        if self._at('IDENT') and self._peek(1).kind == 'in':
            v = self._eat('IDENT').value
            self._eat('in')
            iterable = self._parse_range_or_expr()
            body = self.parse_block()
            return ForInStmt(v, iterable, body, ln)
        # classic C-style
        j, depth, has_semi = self.i, 0, False
        while j < len(self.toks):
            k = self.toks[j].kind
            if k == '{' and depth == 0: break
            if k in ('(','[','<'): depth += 1
            if k in (')',']','>'): depth -= 1
            if k == ';' and depth == 0: has_semi = True; break
            j += 1
        init = cond = post = None
        if has_semi:
            if not self._at(';'):
                init = self.parse_var_decl() if self._at('let') else ExprStmt(self.parse_expr())
            self._eat(';')
            if not self._at(';'): cond = self.parse_expr()
            self._eat(';')
            if not self._at('{'): post = ExprStmt(self.parse_expr())
        else:
            cond = self.parse_expr()
        return ForStmt(init, cond, post, self.parse_block(), ln)

    def _parse_range_or_expr(self) -> Node:
        lo = self.parse_expr()
        if self._at('..') or self._at('..='):
            t = self._peek(); self.i += 1
            hi = self.parse_expr()
            return RangeExpr(lo, hi, t.kind == '..=', t.line)
        return lo

    def parse_spawn(self) -> SpawnStmt:
        ln = self._peek().line
        self._eat('spawn')
        call = self.parse_expr()
        self._maybe(';')
        if not isinstance(call, CallExpr):
            raise AuraError(f"[Parser] {ln}: 'spawn' requires a call expression", ln)
        return SpawnStmt(call, ln)

    def parse_arena(self) -> ArenaStmt:
        ln = self._peek().line
        self._eat('arena')
        return ArenaStmt(self.parse_block(), ln)

    # ---- expressions ----
    def parse_expr(self) -> Node: return self._assign()
    def _assign(self) -> Node:
        l = self._or()
        t = self._peek()
        if t.kind == '=' or t.kind in COMPOUND_OPS:
            op = t.kind; self.i += 1
            return AssignExpr(l, self._assign(), t.line, op)
        return l
    def _or(self) -> Node:
        l = self._and()
        while self._at('||'):
            t = self._eat('||'); l = BinaryExpr('||', l, self._and(), t.line)
        return l
    def _and(self) -> Node:
        l = self._eq()
        while self._at('&&'):
            t = self._eat('&&'); l = BinaryExpr('&&', l, self._eq(), t.line)
        return l
    def _eq(self) -> Node:
        l = self._cmp()
        while self._at('==') or self._at('!='):
            t = self._peek(); self.i += 1
            l = BinaryExpr(t.kind, l, self._cmp(), t.line)
        return l
    def _cmp(self) -> Node:
        l = self._add()
        while self._at('<') or self._at('>') or self._at('<=') or self._at('>=') or self._at('in'):
            t = self._peek(); self.i += 1
            l = BinaryExpr(t.kind, l, self._add(), t.line)
        return l
    def _add(self) -> Node:
        l = self._mul()
        while self._at('+') or self._at('-') or self._at('<-'):
            t = self._peek(); self.i += 1
            if t.kind == '<-': l = ChanSend(l, self._mul(), t.line)
            else: l = BinaryExpr(t.kind, l, self._mul(), t.line)
        return l
    def _mul(self) -> Node:
        l = self._unary()
        while self._at('*') or self._at('/') or self._at('%'):
            t = self._peek(); self.i += 1
            l = BinaryExpr(t.kind, l, self._unary(), t.line)
        return l
    def _unary(self) -> Node:
        t = self._peek()
        if t.kind in ('!','-','*','&'):
            self.i += 1
            return UnaryExpr(t.kind, self._unary(), t.line)
        if t.kind == '<-':
            self.i += 1
            return ChanRecv(self._unary(), t.line)
        return self._postfix()

    def _postfix(self) -> Node:
        e = self._primary()
        while True:
            if self._at('('):
                ln = self._peek().line
                self._eat('(')
                args: List[Node] = []
                if not self._at(')'):
                    args.append(self.parse_expr())
                    while self._maybe(','): args.append(self.parse_expr())
                self._eat(')')
                e = CallExpr(e, args, ln)
            elif self._at('.'):
                ln = self._peek().line
                self._eat('.')
                f = self._eat('IDENT').value
                e = MemberAccess(e, f, ln)
            elif self._at('['):
                ln = self._peek().line
                self._eat('['); idx = self.parse_expr(); self._eat(']')
                e = IndexExpr(e, idx, ln)
            elif self._at('?'):
                ln = self._peek().line
                self._eat('?')
                e = TryExpr(e, ln)
            elif self._at('++') or self._at('--'):
                ln = self._peek().line
                op = self._eat(self._peek().kind).kind
                compound = '+=' if op == '++' else '-='
                e = AssignExpr(e, IntLit(1, ln), ln, op=compound)
            else: break
        return e

    def _primary(self) -> Node:
        t = self._peek()
        if t.kind == 'NUMBER': self._eat('NUMBER'); return IntLit(int(t.value), t.line)
        if t.kind == 'FLOAT':  self._eat('FLOAT');  return FloatLit(float(t.value), t.line)
        if t.kind == 'STRING': self._eat('STRING'); return StringLit(t.value, t.line)
        if t.kind in ('true','false'):
            self._eat(t.kind); return BoolLit(t.kind == 'true', t.line)
        if t.kind == 'nil': self._eat('nil'); return NilLit(t.line)
        if t.kind == 'IDENT':
            self._eat('IDENT'); name = t.value
            if self._at('{'):
                nxt = self._peek(1)
                if nxt.kind == '}' or (nxt.kind == 'IDENT' and self._peek(2).kind == ':'):
                    self._eat('{')
                    inits = []
                    while not self._at('}'):
                        fn = self._eat('IDENT').value
                        self._eat(':')
                        inits.append((fn, self.parse_expr()))
                        if not self._maybe(','): break
                    self._eat('}')
                    return StructLit(name, inits, t.line)
            return Ident(name, t.line)
        if t.kind == 'make':
            self._eat('make'); self._eat('(')
            if self._at('chan'):
                self._eat('chan')
                et = self.parse_type(); self._eat(',')
                cap = self.parse_expr(); self._eat(')')
                return ChanMake(et, cap, t.line)
            elif self._at('map'):
                self._eat('map'); self._eat('[')
                kt = self.parse_type(); self._eat(']')
                vt = self.parse_type(); self._eat(')')
                return MapMake(kt, vt, t.line)
            else:
                raise AuraError(f"[Parser] {t.line}:{t.col}: make expects 'chan' or 'map'", t.line, t.col)
        if t.kind == '[':
            ln = t.line
            self._eat('[')
            elems = []
            if not self._at(']'):
                elems.append(self.parse_expr())
                while self._maybe(','):
                    if self._at(']'): break
                    elems.append(self.parse_expr())
            self._eat(']')
            return SliceLit(elems, ln)
        if t.kind == '(':
            self._eat('('); e = self.parse_expr(); self._eat(')'); return e
        raise AuraError(f"[Parser] {t.line}:{t.col}: unexpected token '{t.value or t.kind}'", t.line, t.col)


# ═══════════════════════════════════════════════════════════════════════════
# 4 — ANALYZER
# ═══════════════════════════════════════════════════════════════════════════

class Scope:
    def __init__(self, parent: Optional[Scope] = None): self.parent, self.symbols = parent, {}
    def define(self, n: str, t: Tuple, kind: Optional[str] = None): self.symbols[n] = (t, kind)
    def lookup(self, n: str) -> Optional[Tuple[Tuple, Optional[str]]]:
        s = self
        while s:
            if n in s.symbols: return s.symbols[n]
            s = s.parent
        return None


class Analyzer:
    def __init__(self):
        self.functions: Dict[str, FuncDecl] = {}
        self.extern_funcs: set = set()
        self.structs: Dict[str, Dict[str, Tuple]] = {}
        self.struct_order: Dict[str, List[str]] = {}
        self.enums: Dict[str, Dict[str, List[Tuple]]] = {}
        self.consts: Dict[str, Tuple] = {}
        self.globals = Scope()
        self.scope = self.globals
        self.current_ret: Optional[TypeRef] = None
        self.arena_depth: int = 0

    def _err(self, ln: int, m: str): raise AuraError(f"[Analyzer] {ln}: {m}", ln)

    def _resolve(self, t: Optional[TypeRef]) -> Tuple:
        if t is None: return ('void',)
        n = t.name
        if n in PRIMITIVES: return (n,)
        if n == 'ptr':   return ('ptr', self._resolve(t.args[0]))
        if n == 'chan':  return ('chan', self._resolve(t.args[0]))
        if n == 'slice': return ('slice', self._resolve(t.args[0]))
        if n == 'map':   return ('map', self._resolve(t.args[0]), self._resolve(t.args[1]))
        if n == 'result':
            return ('result', self._resolve(t.args[0]), self._resolve(t.args[1]))
        if n in self.structs: return ('struct', n)
        if n in self.enums:   return ('enum', n)
        return ('opaque', n)

    def _is_arc(self, t: Tuple) -> Optional[str]:
        if t[0] == 'string': return 'string'
        if t[0] == 'slice':  return 'slice'
        if t[0] == 'map':    return 'map'
        return None

    def analyze(self, prog: Program):
        # 1. Enums
        for d in prog.decls:
            if isinstance(d, EnumDecl):
                if d.name in self.enums: self._err(d.line, f"enum '{d.name}' already declared")
                self.enums[d.name] = {}
                for vn, vtypes in d.variants:
                    self.enums[d.name][vn] = [self._resolve(vt) for vt in vtypes]
        # 2. Structs
        for d in prog.decls:
            if isinstance(d, StructDecl):
                if d.name in self.structs: self._err(d.line, f"struct '{d.name}' already declared")
                self.structs[d.name] = {}
                self.struct_order[d.name] = [fn for fn, _ in d.fields]
        for d in prog.decls:
            if isinstance(d, StructDecl):
                for fn, ft in d.fields:
                    self.structs[d.name][fn] = self._resolve(ft)
        # 3. Functions
        for d in prog.decls:
            if isinstance(d, FuncDecl):
                key = f"{d.receiver.type.name}.{d.name}" if d.receiver and d.receiver.type.name != 'ptr' else (
                      f"{d.receiver.type.args[0].name}.{d.name}" if d.receiver else d.name)
                if key in self.functions: self._err(d.line, f"'{key}' already declared")
                self.functions[key] = d
            elif isinstance(d, ExternBlock):
                for f in d.decls:
                    if f.name in self.functions: self._err(f.line, f"'{f.name}' already declared")
                    self.functions[f.name] = f
                    self.extern_funcs.add(f.name)
        # 4. Consts
        for d in prog.decls:
            if isinstance(d, ConstDecl):
                t = self._analyze_expr(d.init)
                self.consts[d.name] = (t, d.init)
        # 5. Bodies
        for d in prog.decls:
            if isinstance(d, FuncDecl) and not d.is_extern:
                self._analyze_func(d)

    def _analyze_func(self, f: FuncDecl):
        self.scope = Scope(self.globals)
        self.current_ret = f.ret_type
        if f.receiver:
            rt = self._resolve(f.receiver.type)
            self.scope.define(f.receiver.name, rt, self._is_arc(rt))
        for p in f.params:
            if p.variadic: continue
            pt = self._resolve(p.type)
            self.scope.define(p.name, pt, self._is_arc(pt))
        self._analyze_block(f.body, new_scope=False)

    def _analyze_block(self, b: Block, new_scope: bool = True):
        if new_scope: self.scope = Scope(self.scope)
        for s in b.stmts: self._analyze_stmt(s)
        if new_scope: self.scope = self.scope.parent

    def _analyze_stmt(self, s: Node):
        if isinstance(s, VarDecl):
            it = self._analyze_expr(s.init) if s.init else None
            dt = self._resolve(s.type) if s.type else it
            if dt is None: self._err(s.line, f"'{s.name}' declared without type or initializer")
            if it and not self._type_eq(dt, it) and not (dt == ('float',) and it == ('int',)):
                self._err(s.line, f"type mismatch for '{s.name}': {dt} vs {it}")
            self.scope.define(s.name, dt, self._is_arc(dt))
            s._resolved_type = dt
            return

        if isinstance(s, DeferStmt):
            self._analyze_stmt(s.stmt)
            return

        if isinstance(s, ReturnStmt):
            rt = self._resolve(self.current_ret) if self.current_ret else ('void',)
            if s.value is None:
                if rt != ('void',): self._err(s.line, f"function must return {rt}")
            else:
                vt = self._analyze_expr(s.value)
                if rt[0] == 'result' and vt[0] == 'result': pass
                elif not self._type_eq(rt, vt) and not (rt == ('float',) and vt == ('int',)):
                    self._err(s.line, f"return mismatch: expected {rt}, got {vt}")
            return

        if isinstance(s, IfStmt):
            ct = self._analyze_expr(s.cond)
            if ct != ('bool',): self._err(s.line, f"'if' condition must be bool, got {ct}")
            self._analyze_block(s.then)
            if s.els:
                if isinstance(s.els, IfStmt): self._analyze_stmt(s.els)
                else: self._analyze_block(s.els)
            return

        if isinstance(s, ForStmt):
            self.scope = Scope(self.scope)
            if isinstance(s.init, ExprStmt) and isinstance(s.init.expr, AssignExpr) \
               and isinstance(s.init.expr.target, Ident):
                nm = s.init.expr.target.name
                if self.scope.lookup(nm) is None:
                    vt = self._analyze_expr(s.init.expr.value)
                    self.scope.define(nm, vt, self._is_arc(vt))
                    s.init.expr.target._auto_declared = True
                else:
                    self._analyze_expr(s.init.expr)
            elif s.init is not None:
                self._analyze_stmt(s.init)
            if s.cond is not None:
                ct = self._analyze_expr(s.cond)
                if ct != ('bool',): self._err(s.line, f"'for' condition must be bool, got {ct}")
            if s.post is not None: self._analyze_stmt(s.post)
            self._analyze_block(s.body)
            self.scope = self.scope.parent
            return

        if isinstance(s, ForInStmt):
            self.scope = Scope(self.scope)
            if isinstance(s.iterable, RangeExpr):
                lt = self._analyze_expr(s.iterable.lo)
                ht = self._analyze_expr(s.iterable.hi)
                if lt != ('int',) or ht != ('int',):
                    self._err(s.line, f"range bounds must be int, got {lt}, {ht}")
                s.kind, s.elem_type, s.inclusive = 'range', ('int',), s.iterable.inclusive
                s._resolved_iter = ('range',)
            else:
                it = self._analyze_expr(s.iterable)
                if it[0] == 'slice': s.kind, s.elem_type = 'slice', it[1]
                elif it[0] == 'chan': s.kind, s.elem_type = 'chan', it[1]
                else: self._err(s.line, f"cannot iterate over {it}")
                s._resolved_iter = it
            self.scope.define(s.var, s.elem_type, self._is_arc(s.elem_type))
            self._analyze_block(s.body)
            self.scope = self.scope.parent
            return

        if isinstance(s, ArenaStmt):
            self.arena_depth += 1
            self._analyze_block(s.body)
            self.arena_depth -= 1
            return

        if isinstance(s, SpawnStmt):
            if not isinstance(s.call.func, Ident): self._err(s.line, "'spawn' requires a named function")
            if s.call.func.name not in self.functions: self._err(s.line, f"spawned func '{s.call.func.name}' undefined")
            self._analyze_expr(s.call)
            return

        if isinstance(s, Block): self._analyze_block(s); return
        if isinstance(s, ExprStmt): self._analyze_expr(s.expr); return
        if isinstance(s, (BreakStmt, ContinueStmt)): return
        self._err(getattr(s,'line',0), f"unhandled statement {type(s).__name__}")

    def _type_eq(self, a: Tuple, b: Tuple) -> bool: return a == b

    def _analyze_expr(self, e: Node) -> Tuple:
        if isinstance(e, IntLit):    return ('int',)
        if isinstance(e, FloatLit):  return ('float',)
        if isinstance(e, BoolLit):   return ('bool',)
        if isinstance(e, StringLit): return ('string',)
        if isinstance(e, NilLit):    return ('ptr', ('void',))

        if isinstance(e, Ident):
            hit = self.scope.lookup(e.name)
            if hit is None:
                if e.name in self.consts: return self.consts[e.name][0]
                if e.name in self.functions: return ('func', e.name)
                # Enum variant?
                for ename, evars in self.enums.items():
                    if e.name in evars and not evars[e.name]:
                        return ('enum', ename)
                self._err(e.line, f"undeclared identifier '{e.name}'")
            return hit[0]

        if isinstance(e, BinaryExpr):
            lt = self._analyze_expr(e.left)
            rt = self._analyze_expr(e.right)
            if e.op in ('==','!=','<','>','<=','>='):
                if e.op in ('==','!=') and lt == ('string',) and rt == ('string',): return ('bool',)
                return ('bool',)
            if e.op in ('&&','||'): return ('bool',)
            if e.op == '+':
                if lt == ('string',) and rt == ('string',): return ('string',)
                if lt[0] == 'ptr' and rt == ('int',): return lt
            if e.op == 'in':
                if rt[0] == 'map' and self._type_eq(rt[1], lt): return ('bool',)
                self._err(e.line, f"'in' expects element and map, got {lt} and {rt}")
            if lt == rt: return lt
            if {lt[0], rt[0]} <= {'int','float'}: return ('float',)
            self._err(e.line, f"'{e.op}' between {lt} and {rt}")

        if isinstance(e, UnaryExpr):
            t = self._analyze_expr(e.operand)
            if e.op == '!': return ('bool',)
            if e.op == '-': return t
            if e.op == '&': return ('ptr', t)
            if e.op == '*':
                if t[0] != 'ptr': self._err(e.line, f"'*' dereference needs pointer, got {t}")
                return t[1]
            self._err(e.line, f"unhandled unary '{e.op}'")

        if isinstance(e, AssignExpr):
            tt = self._analyze_expr(e.target)
            vt = self._analyze_expr(e.value)
            if e.op == '=':
                if not self._type_eq(tt, vt) and not (tt == ('float',) and vt == ('int',)):
                    self._err(e.line, f"assignment mismatch: {tt} = {vt}")
            return tt

        if isinstance(e, CallExpr):
            # Universal Method Resolution
            if isinstance(e.func, MemberAccess):
                base_t = self._analyze_expr(e.func.obj)
                st = None
                if base_t[0] == 'struct': st = base_t[1]
                elif base_t[0] == 'ptr' and base_t[1][0] == 'struct': st = base_t[1][1]
                if st and f"{st}.{e.func.field}" in self.functions:
                    return self._analyze_method_call(e, st, e.func.field)

            if not isinstance(e.func, Ident): self._err(e.line, "indirect function call unsupported")
            name = e.func.name

            # Intrinsics
            if name in ('print','println'):
                self._analyze_expr(e.args[0]); e._intrinsic = name; return ('void',)
            if name == 'len':
                at = self._analyze_expr(e.args[0])
                if at[0] not in ('string','slice','map'): self._err(e.line, "'len' expects string, slice, or map")
                e._intrinsic = 'len'; return ('int',)
            if name == 'panic':
                self._analyze_expr(e.args[0]); e._intrinsic = 'panic'; return ('void',)
            if name == 'ok':
                vt = self._analyze_expr(e.args[0]); e._intrinsic = 'ok'; return ('result', vt, ('int',))
            if name == 'err':
                et = self._analyze_expr(e.args[0]); e._intrinsic = 'err'; return ('result', ('void',), et)
            if name == 'close':
                ct = self._analyze_expr(e.args[0]); e._intrinsic = 'close'; return ('void',)

            fn = self.functions.get(name)
            if fn is None: self._err(e.line, f"call to undefined function '{name}'")
            e._is_extern = name in self.extern_funcs
            for a in e.args: self._analyze_expr(a)
            rt = self._resolve(fn.ret_type) if fn.ret_type else ('void',)
            e._resolved_ret = rt
            return rt

        if isinstance(e, ChanSend):
            ct = self._analyze_expr(e.chan); vt = self._analyze_expr(e.value)
            e._elem = ct[1]
            return ('void',)

        if isinstance(e, ChanRecv):
            ct = self._analyze_expr(e.chan)
            e._resolved_type = ct[1]
            return ct[1]

        if isinstance(e, ChanMake):
            et = self._resolve(e.elem_type); self._analyze_expr(e.capacity)
            e._elem_type = et
            return ('chan', et)

        if isinstance(e, MapMake):
            kt = self._resolve(e.key_type); vt = self._resolve(e.val_type)
            e._key_type, e._val_type = kt, vt
            return ('map', kt, vt)

        if isinstance(e, MemberAccess):
            ot = self._analyze_expr(e.obj)
            st = ot[1][1] if (ot[0] == 'ptr' and ot[1][0] == 'struct') else (ot[1] if ot[0] == 'struct' else None)
            if not st or st not in self.structs: self._err(e.line, f"member access '.{e.field}' on non-struct {ot}")
            t = self.structs[st][e.field]
            e._resolved_type = t
            return t

        if isinstance(e, IndexExpr):
            ot = self._analyze_expr(e.obj); it = self._analyze_expr(e.index)
            if ot[0] == 'slice': e._resolved_type = ot[1]; return ot[1]
            if ot[0] == 'ptr':   e._resolved_type = ot[1]; return ot[1]
            if ot[0] == 'map':
                if not self._type_eq(ot[1], it): self._err(e.line, f"map key mismatch: expected {ot[1]}, got {it}")
                e._resolved_type = ot[2]; return ot[2]
            self._err(e.line, f"cannot index type {ot}")

        if isinstance(e, SliceLit):
            et = self._analyze_expr(e.elements[0])
            for el in e.elements[1:]: self._analyze_expr(el)
            e._elem_type = et
            return ('slice', et)

        if isinstance(e, StructLit):
            fmap = self.structs[e.name]
            for fn, fe in e.inits: self._analyze_expr(fe)
            e._resolved_type = ('struct', e.name)
            return ('struct', e.name)

        if isinstance(e, TryExpr):
            t = self._analyze_expr(e.expr)
            if t[0] != 'result': self._err(e.line, f"'?' operator requires result type, got {t}")
            e._resolved_type = t[1]
            return t[1]

        self._err(getattr(e,'line',0), f"unhandled expr {type(e).__name__}")

    def _analyze_method_call(self, e: CallExpr, st: str, mname: str) -> Tuple:
        key = f"{st}.{mname}"
        fn = self.functions[key]
        for a in e.args: self._analyze_expr(a)
        e._is_method = True
        e._method_key = key
        e._method_recv = e.func.obj
        rt = self._resolve(fn.ret_type) if fn.ret_type else ('void',)
        e._resolved_ret = rt
        return rt


# ═══════════════════════════════════════════════════════════════════════════
# 5 — RUNTIME HEADER
# ═══════════════════════════════════════════════════════════════════════════

RUNTIME_HEADER = r'''
#ifndef AURA_RT_H
#define AURA_RT_H
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <stdatomic.h>
#include <sched.h>
#include <unistd.h>
#include <time.h>

/* ───── ARC + Arenas ───── */
typedef struct { _Atomic int64_t rc; int64_t flags; } AuraHeader;

typedef struct AuraArenaBlock { struct AuraArenaBlock* next; size_t used, cap; char data[]; } AuraArenaBlock;
typedef struct { AuraArenaBlock* head; size_t default_block; } AuraArena;
static _Thread_local AuraArena* __aura_cur_arena = NULL;

static inline void* aura_arena_alloc(AuraArena* a, size_t size) {
    size = (size + 15) & ~((size_t)15);
    if (!a->head || a->head->used + size > a->head->cap) {
        size_t cap = size > a->default_block ? size : a->default_block;
        AuraArenaBlock* b = (AuraArenaBlock*)malloc(sizeof(AuraArenaBlock) + cap);
        if (!b) { fprintf(stderr, "aura: arena OOM\n"); abort(); }
        b->next = a->head; b->used = 0; b->cap = cap; a->head = b;
    }
    void* p = a->head->data + a->head->used; a->head->used += size; return p;
}
static inline void aura_arena_init(AuraArena* a, size_t b) {
    a->head = NULL; a->default_block = b ? b : ((size_t)1 << 16);
}
static inline void aura_arena_free(AuraArena* a) {
    AuraArenaBlock* b = a->head;
    while (b) { AuraArenaBlock* n = b->next; free(b); b = n; }
    a->head = NULL;
}
static inline void* aura_alloc(size_t size) {
    if (__aura_cur_arena) {
        AuraHeader* h = (AuraHeader*)aura_arena_alloc(__aura_cur_arena, size + sizeof(AuraHeader));
        h->rc = 1; h->flags = 1;
        return (char*)h + sizeof(AuraHeader);
    }
    AuraHeader* h = (AuraHeader*)malloc(size + sizeof(AuraHeader));
    if (!h) { fprintf(stderr, "aura: OOM\n"); abort(); }
    h->rc = 1; h->flags = 0;
    return (char*)h + sizeof(AuraHeader);
}
static inline void aura_retain(void* p) {
    if (!p) return; AuraHeader* h = (AuraHeader*)p - 1;
    if (h->flags & 1) return;
    atomic_fetch_add_explicit(&h->rc, 1, memory_order_relaxed);
}
static inline void aura_release(void* p) {
    if (!p) return; AuraHeader* h = (AuraHeader*)p - 1;
    if (h->flags & 1) return;
    if (atomic_fetch_sub_explicit(&h->rc, 1, memory_order_acq_rel) == 1) free(h);
}
static inline void aura_retain_local(void* p) {
    if (!p) return; AuraHeader* h = (AuraHeader*)p - 1;
    if (h->flags & 1) return; h->rc++;
}
static inline void aura_release_local(void* p) {
    if (!p) return; AuraHeader* h = (AuraHeader*)p - 1;
    if (h->flags & 1) return; if (--h->rc == 0) free(h);
}

/* ───── Strings ───── */
typedef struct { size_t len; char data[]; } AuraString;

static inline AuraString* aura_string_new(const char* s) {
    size_t n = strlen(s);
    AuraString* o = (AuraString*)aura_alloc(sizeof(AuraString) + n + 1);
    o->len = n; memcpy(o->data, s, n); o->data[n] = '\0'; return o;
}
static inline AuraString* aura_string_concat(AuraString* a, AuraString* b) {
    size_t n = a->len + b->len;
    AuraString* o = (AuraString*)aura_alloc(sizeof(AuraString) + n + 1);
    o->len = n;
    memcpy(o->data, a->data, a->len);
    memcpy(o->data + a->len, b->data, b->len);
    o->data[n] = '\0'; return o;
}
static inline bool aura_string_eq(AuraString* a, AuraString* b) {
    if (a == b) return true;
    if (!a || !b) return false;
    if (a->len != b->len) return false;
    return memcmp(a->data, b->data, a->len) == 0;
}

/* ───── Slices ───── */
typedef struct { void* data; int64_t len; int64_t cap; } AuraSlice;

/* ───── Hash Maps ───── */
typedef struct { int64_t key; int64_t val; bool occupied; } AuraMapEntry;
typedef struct {
    AuraMapEntry* entries;
    size_t cap, count;
    bool is_str_key, is_str_val;
} AuraMap;

static inline uint64_t aura_hash_key(int64_t k, bool is_str) {
    if (!is_str) return (uint64_t)k * 11400714819323198485ULL;
    AuraString* s = (AuraString*)k;
    uint64_t h = 14695981039346656037ULL;
    for (size_t i = 0; i < s->len; i++) { h ^= (uint8_t)s->data[i]; h *= 1099511628211ULL; }
    return h;
}
static inline AuraMap* aura_map_new(bool str_k, bool str_v) {
    AuraMap* m = (AuraMap*)aura_alloc(sizeof(AuraMap));
    m->cap = 16; m->count = 0;
    m->is_str_key = str_k; m->is_str_val = str_v;
    m->entries = (AuraMapEntry*)calloc(m->cap, sizeof(AuraMapEntry));
    return m;
}
static inline void aura_map_set(AuraMap* m, int64_t k, int64_t v) {
    if (m->count * 2 >= m->cap) {
        size_t old_cap = m->cap;
        AuraMapEntry* old_entries = m->entries;
        m->cap *= 2;
        m->entries = (AuraMapEntry*)calloc(m->cap, sizeof(AuraMapEntry));
        m->count = 0;
        for (size_t i = 0; i < old_cap; i++) {
            if (old_entries[i].occupied)
                aura_map_set(m, old_entries[i].key, old_entries[i].val);
        }
        free(old_entries);
    }
    uint64_t idx = aura_hash_key(k, m->is_str_key) & (m->cap - 1);
    while (m->entries[idx].occupied) {
        if (m->is_str_key) {
            if (aura_string_eq((AuraString*)m->entries[idx].key, (AuraString*)k)) break;
        } else if (m->entries[idx].key == k) break;
        idx = (idx + 1) & (m->cap - 1);
    }
    if (!m->entries[idx].occupied) { m->count++; m->entries[idx].occupied = true; }
    m->entries[idx].key = k; m->entries[idx].val = v;
}
static inline int64_t aura_map_get(AuraMap* m, int64_t k) {
    uint64_t idx = aura_hash_key(k, m->is_str_key) & (m->cap - 1);
    while (m->entries[idx].occupied) {
        if (m->is_str_key) {
            if (aura_string_eq((AuraString*)m->entries[idx].key, (AuraString*)k)) return m->entries[idx].val;
        } else if (m->entries[idx].key == k) return m->entries[idx].val;
        idx = (idx + 1) & (m->cap - 1);
    }
    return 0;
}
static inline bool aura_map_contains(AuraMap* m, int64_t k) {
    uint64_t idx = aura_hash_key(k, m->is_str_key) & (m->cap - 1);
    while (m->entries[idx].occupied) {
        if (m->is_str_key) {
            if (aura_string_eq((AuraString*)m->entries[idx].key, (AuraString*)k)) return true;
        } else if (m->entries[idx].key == k) return true;
        idx = (idx + 1) & (m->cap - 1);
    }
    return false;
}

/* ───── Channels ───── */
typedef struct AuraChanNode { struct AuraChanNode* next; char data[]; } AuraChanNode;
typedef struct {
    size_t elem_size, capacity, count;
    AuraChanNode *head, *tail;
    int closed;
    pthread_mutex_t mu;
    pthread_cond_t  not_empty, not_full;
} AuraChan;

static inline AuraChan* aura_chan_new(size_t es, size_t cap) {
    AuraChan* c = (AuraChan*)calloc(1, sizeof(AuraChan));
    c->elem_size = es; c->capacity = cap;
    pthread_mutex_init(&c->mu, NULL);
    pthread_cond_init(&c->not_empty, NULL);
    pthread_cond_init(&c->not_full, NULL);
    return c;
}
static inline void aura_chan_send(AuraChan* c, const void* val) {
    pthread_mutex_lock(&c->mu);
    while (c->capacity > 0 && c->count >= c->capacity && !c->closed)
        pthread_cond_wait(&c->not_full, &c->mu);
    if (c->closed) { pthread_mutex_unlock(&c->mu); return; }
    AuraChanNode* n = (AuraChanNode*)malloc(sizeof(AuraChanNode) + c->elem_size);
    memcpy(n->data, val, c->elem_size);
    n->next = NULL;
    if (c->tail) c->tail->next = n; else c->head = n;
    c->tail = n; c->count++;
    pthread_cond_signal(&c->not_empty);
    pthread_mutex_unlock(&c->mu);
}
static inline bool aura_chan_recv_ok(AuraChan* c, void* out) {
    pthread_mutex_lock(&c->mu);
    while (c->count == 0 && !c->closed)
        pthread_cond_wait(&c->not_empty, &c->mu);
    if (c->count == 0 && c->closed) { pthread_mutex_unlock(&c->mu); return false; }
    AuraChanNode* n = c->head;
    c->head = n->next;
    if (!c->head) c->tail = NULL;
    c->count--;
    memcpy(out, n->data, c->elem_size);
    free(n);
    pthread_cond_signal(&c->not_full);
    pthread_mutex_unlock(&c->mu);
    return true;
}
static inline void aura_chan_recv(AuraChan* c, void* out) {
    if (!aura_chan_recv_ok(c, out)) memset(out, 0, c->elem_size);
}
static inline void aura_chan_close(AuraChan* c) {
    pthread_mutex_lock(&c->mu);
    c->closed = 1;
    pthread_cond_broadcast(&c->not_empty);
    pthread_cond_broadcast(&c->not_full);
    pthread_mutex_unlock(&c->mu);
}

/* ───── M:N Scheduler (Turn-Sequenced Lock-Free Ring Buffer) ───── */
typedef void* (*AuraTaskFn)(void*);
#define AURA_Q_BITS 16
#define AURA_Q_SIZE (1u << AURA_Q_BITS)
#define AURA_Q_MASK (AURA_Q_SIZE - 1)

typedef struct {
    _Atomic size_t turn;
    AuraTaskFn     fn;
    void*          args;
} AuraSlot;

typedef struct {
    AuraSlot        slots[AURA_Q_SIZE];
    _Atomic size_t  tail;
    _Atomic size_t  head;
    pthread_t*      workers;
    int             n_workers;
    _Atomic bool    running;
    pthread_mutex_t sleep_mu;
    pthread_cond_t  sleep_cv;
} AuraPool;

static AuraPool __aura_pool;
static pthread_once_t __aura_pool_once = PTHREAD_ONCE_INIT;

static void* __aura_worker(void* arg) {
    (void)arg;
    while (atomic_load_explicit(&__aura_pool.running, memory_order_acquire)) {
        size_t h = atomic_load_explicit(&__aura_pool.head, memory_order_relaxed);
        AuraSlot* slot = &__aura_pool.slots[h & AURA_Q_MASK];
        size_t turn = atomic_load_explicit(&slot->turn, memory_order_acquire);
        if (turn == (h * 2 + 1)) {
            if (atomic_compare_exchange_weak_explicit(&__aura_pool.head, &h, h + 1, memory_order_acq_rel, memory_order_relaxed)) {
                AuraTaskFn fn = slot->fn;
                void* args = slot->args;
                atomic_store_explicit(&slot->turn, (h + AURA_Q_SIZE) * 2, memory_order_release);
                fn(args);
                free(args);
            }
            continue;
        }
        pthread_mutex_lock(&__aura_pool.sleep_mu);
        struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_nsec += 200000;
        if (ts.tv_nsec >= 1000000000) { ts.tv_sec++; ts.tv_nsec -= 1000000000; }
        pthread_cond_timedwait(&__aura_pool.sleep_cv, &__aura_pool.sleep_mu, &ts);
        pthread_mutex_unlock(&__aura_pool.sleep_mu);
    }
    return NULL;
}
static void __aura_pool_init(void) {
    for (size_t i = 0; i < AURA_Q_SIZE; i++) atomic_store(&__aura_pool.slots[i].turn, i * 2);
    atomic_store(&__aura_pool.tail, 0); atomic_store(&__aura_pool.head, 0);
    atomic_store(&__aura_pool.running, true);
    pthread_mutex_init(&__aura_pool.sleep_mu, NULL);
    pthread_cond_init(&__aura_pool.sleep_cv, NULL);
    long nc = sysconf(_SC_NPROCESSORS_ONLN);
    if (nc < 2) nc = 2; if (nc > 64) nc = 64;
    __aura_pool.n_workers = (int)nc;
    __aura_pool.workers = (pthread_t*)malloc(sizeof(pthread_t) * nc);
    for (int i = 0; i < nc; i++) {
        pthread_create(&__aura_pool.workers[i], NULL, __aura_worker, NULL);
        pthread_detach(__aura_pool.workers[i]);
    }
}
static inline void aura_spawn(AuraTaskFn fn, const void* args, size_t args_size) {
    pthread_once(&__aura_pool_once, __aura_pool_init);
    void* boxed = malloc(args_size ? args_size : 1);
    if (args_size) memcpy(boxed, args, args_size);
    for (;;) {
        size_t t = atomic_load_explicit(&__aura_pool.tail, memory_order_relaxed);
        AuraSlot* slot = &__aura_pool.slots[t & AURA_Q_MASK];
        size_t turn = atomic_load_explicit(&slot->turn, memory_order_acquire);
        if (turn == t * 2) {
            if (atomic_compare_exchange_weak_explicit(&__aura_pool.tail, &t, t + 1, memory_order_acq_rel, memory_order_relaxed)) {
                slot->fn = fn; slot->args = boxed;
                atomic_store_explicit(&slot->turn, t * 2 + 1, memory_order_release);
                break;
            }
        } else { sched_yield(); }
    }
    pthread_mutex_lock(&__aura_pool.sleep_mu);
    pthread_cond_signal(&__aura_pool.sleep_cv);
    pthread_mutex_unlock(&__aura_pool.sleep_mu);
}

/* ───── Standard Library Primitives ───── */
static inline void aura_print_int(int64_t x)    { printf("%ld\n", (long)x); }
static inline void aura_print_float(double x)   { printf("%g\n", x); }
static inline void aura_print_bool(bool b)      { printf("%s\n", b ? "true" : "false"); }
static inline void aura_print_str(AuraString* s){ printf("%s\n", s ? s->data : "(null)"); }
static inline void aura_panic(AuraString* s)    { fprintf(stderr, "aura panic: %s\n", s ? s->data : "(null)"); abort(); }
static inline void aura_panic_cstr(const char* s){ fprintf(stderr, "aura panic: %s\n", s); abort(); }

static inline int64_t aura_now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
    return (int64_t)ts.tv_sec * 1000 + (ts.tv_nsec / 1000000);
}
static inline void aura_sleep_ms(int64_t ms) {
    usleep((useconds_t)ms * 1000);
}
static inline AuraString* aura_read_file(AuraString* path) {
    FILE* f = fopen(path->data, "rb");
    if (!f) return aura_string_new("");
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    AuraString* s = (AuraString*)aura_alloc(sizeof(AuraString) + sz + 1);
    s->len = sz;
    fread(s->data, 1, sz, f);
    s->data[sz] = '\0';
    fclose(f);
    return s;
}
static inline bool aura_write_file(AuraString* path, AuraString* content) {
    FILE* f = fopen(path->data, "wb");
    if (!f) return false;
    fwrite(content->data, 1, content->len, f);
    fclose(f);
    return true;
}

#endif /* AURA_RT_H */
'''


# ═══════════════════════════════════════════════════════════════════════════
# 6 — CODE GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

class CCodegen:
    def __init__(self, an: Analyzer):
        self.an = an
        self.out: List[str] = []
        self.indent = 0
        self.var_types: Dict[str, Tuple] = {}
        self.scope_stack: List[Dict[str, Tuple]] = []
        self.arc_stack: List[List[Tuple[str, Optional[str]]]] = []
        self.defer_stack: List[List[Node]] = []
        self.tmp = 0
        self.arena_n = 0
        self.spawn_n = 0
        self.spawn_helpers: List[str] = []

    def _w(self, s: str = ''): self.out.append('    ' * self.indent + s)
    def _fresh(self, p: str = 't') -> str: self.tmp += 1; return f"__{p}{self.tmp}"

    def _push_scope(self):
        self.scope_stack.append(dict(self.var_types))
        self.arc_stack.append([])
        self.defer_stack.append([])

    def _close_scope(self):
        # 1. Execute defers in LIFO reverse order
        for d in reversed(self.defer_stack[-1]):
            self._emit_stmt(d)
        self.defer_stack.pop()

        # 2. Release local ARC allocations
        for n, k in self.arc_stack[-1]:
            if k == 'string': self._w(f"aura_release_local({n});")
            elif k == 'slice': self._w(f"aura_release_local(({n}).data);")
            elif k == 'map':   self._w(f"aura_release_local({n});")
        self.arc_stack.pop()
        self.var_types = self.scope_stack.pop()

    def _declare(self, name: str, t: Tuple, kind: Optional[str]):
        self.var_types[name] = t
        if self.arc_stack:
            self.arc_stack[-1].append((name, kind))

    def ctype(self, t: Tuple) -> str:
        if t is None: return 'void'
        k = t[0]
        if k == 'int':    return 'int64_t'
        if k == 'float':  return 'double'
        if k == 'bool':   return 'bool'
        if k == 'string': return 'AuraString*'
        if k == 'void':   return 'void'
        if k == 'ptr':    return self.ctype(t[1]) + '*'
        if k == 'chan':   return 'AuraChan*'
        if k == 'slice':  return 'AuraSlice'
        if k == 'map':    return 'AuraMap*'
        if k == 'struct': return 'Aura_' + t[1]
        if k == 'enum':   return 'Aura_' + t[1]
        if k == 'result': return 'AuraResult'
        return 'void*'

    def emit(self, prog: Program) -> str:
        self._w('#include "aura_rt.h"')
        self._w()
        for d in prog.decls:
            if isinstance(d, StructDecl): self._emit_struct(d)
            elif isinstance(d, EnumDecl): self._emit_enum(d)
        self._emit_result_type()
        for d in prog.decls:
            if isinstance(d, ConstDecl): self._emit_const(d)
        for d in prog.decls:
            if isinstance(d, FuncDecl): self._emit_sig(d, forward=True)
            elif isinstance(d, ExternBlock):
                for f in d.decls: self._emit_sig(f, forward=True, extern=True)
        self._w()
        for d in prog.decls:
            if isinstance(d, FuncDecl) and not d.is_extern: self._emit_func(d)
        for h in self.spawn_helpers: self.out.append(h)
        return '\n'.join(self.out)

    def _emit_struct(self, d: StructDecl):
        self._w(f"typedef struct Aura_{d.name} Aura_{d.name};")
        self._w(f"struct Aura_{d.name} {{")
        self.indent += 1
        for fn, ft in d.fields:
            self._w(f"{self.ctype(self.an._resolve(ft))} {fn};")
        self.indent -= 1
        self._w("};")
        self._w()

    def _emit_enum(self, d: EnumDecl):
        self._w(f"typedef struct Aura_{d.name} Aura_{d.name};")
        self._w(f"struct Aura_{d.name} {{")
        self.indent += 1
        self._w("int tag;")
        self._w("union {")
        self.indent += 1
        for vn, vt in d.variants:
            if vt:
                parts = [f"{self.ctype(t)} _{i};" for i, t in enumerate(vt)]
                self._w(f"struct {{ {' '.join(parts)} }} {vn};")
        self.indent -= 1
        self._w("} data;")
        self.indent -= 1
        self._w("};")
        self._w()

    def _emit_result_type(self):
        self._w("typedef struct { bool ok; int64_t err_code;")
        self._w("    union { int64_t i; double f; void* p; } val; } AuraResult;")
        self._w()

    def _emit_const(self, d: ConstDecl):
        t = self.an.consts[d.name][0]
        ct = self.ctype(t)
        self._w(f"static const {ct} {d.name} = {self._emit_expr(d.init)};")

    def _emit_sig(self, f: FuncDecl, forward=False, extern=False):
        ps = []
        if f.receiver:
            rt = self.an._resolve(f.receiver.type)
            ps.append(f"{self.ctype(rt)} {f.receiver.name}")
        for p in f.params:
            if p.variadic: ps.append('...')
            else: ps.append(f"{self.ctype(self.an._resolve(p.type))} {p.name}")
        rt = self.ctype(self.an._resolve(f.ret_type)) if f.ret_type else 'void'
        ps_s = ', '.join(ps) if ps else 'void'
        name = f"Aura_{f.receiver.type.name if f.receiver.type.name != 'ptr' else f.receiver.type.args[0].name}_{f.name}" if f.receiver else f.name
        suffix = ';' if (forward or extern) else ''
        self._w(f"{rt} {name}({ps_s}){suffix}")

    def _emit_func(self, f: FuncDecl):
        self.var_types = {}
        self.scope_stack = [dict()]
        self.arc_stack = [[]]
        self.defer_stack = [[]]
        ps = []
        if f.receiver:
            rt = self.an._resolve(f.receiver.type)
            ps.append(f"{self.ctype(rt)} {f.receiver.name}")
            self.var_types[f.receiver.name] = rt
        for p in f.params:
            ps.append(f"{self.ctype(self.an._resolve(p.type))} {p.name}")
            self.var_types[p.name] = self.an._resolve(p.type)
        rt = self.ctype(self.an._resolve(f.ret_type)) if f.ret_type else 'void'
        ps_s = ', '.join(ps) if ps else 'void'
        name = f"Aura_{f.receiver.type.name if f.receiver.type.name != 'ptr' else f.receiver.type.args[0].name}_{f.name}" if f.receiver else f.name
        self._w(f"{rt} {name}({ps_s}) {{")
        self.indent += 1
        for st in f.body.stmts: self._emit_stmt(st)
        self.indent -= 1
        self._w('}')
        self._w()

    def _emit_stmt(self, s: Node):
        if isinstance(s, VarDecl):      self._emit_var(s)
        elif isinstance(s, DeferStmt):
            self.defer_stack[-1].append(s.stmt)
        elif isinstance(s, ReturnStmt): self._emit_return(s)
        elif isinstance(s, IfStmt):     self._emit_if(s)
        elif isinstance(s, ForStmt):    self._emit_for(s)
        elif isinstance(s, ForInStmt):  self._emit_for_in(s)
        elif isinstance(s, Block):
            self._push_scope(); self._w('{'); self.indent += 1
            for x in s.stmts: self._emit_stmt(x)
            self._close_scope(); self.indent -= 1; self._w('}')
        elif isinstance(s, ExprStmt):
            c = self._emit_expr(s.expr)
            if c.strip(): self._w(f"{c};")
        elif isinstance(s, SpawnStmt): self._emit_spawn(s)
        elif isinstance(s, ArenaStmt): self._emit_arena(s)
        elif isinstance(s, BreakStmt): self._w('break;')
        elif isinstance(s, ContinueStmt): self._w('continue;')

    def _emit_var(self, s: VarDecl):
        t = getattr(s, '_resolved_type', None) or (self.an._resolve(s.type) if s.type else ('int',))
        ct = self.ctype(t)
        if s.init is not None:
            ic = self._emit_expr(s.init)
            self._w(f"{ct} {s.name} = {ic};")
            if t[0] == 'string' and self._needs_retain(s.init):
                self._w(f"aura_retain_local({s.name});")
            elif t[0] == 'slice' and self._needs_retain(s.init):
                self._w(f"aura_retain_local(({s.name}).data);")
            elif t[0] == 'map' and self._needs_retain(s.init):
                self._w(f"aura_retain_local({s.name});")
        else:
            self._w(f"{ct} {s.name} = ({ct}){{0}};")
        self._declare(s.name, t, self.an._is_arc(t))

    def _needs_retain(self, e: Node) -> bool:
        if isinstance(e, (Ident, MemberAccess, IndexExpr)): return True
        return False

    def _emit_return(self, s: ReturnStmt):
        # Precise Return: Evaluate, conditionally retain, unwind defers & ARC, return
        if s.value is None:
            for dlist in reversed(self.defer_stack):
                for d in reversed(dlist): self._emit_stmt(d)
            for sc in reversed(self.arc_stack):
                for n, k in sc:
                    if k == 'string': self._w(f"aura_release_local({n});")
                    elif k == 'slice': self._w(f"aura_release_local(({n}).data);")
                    elif k == 'map':   self._w(f"aura_release_local({n});")
            self._w('return;')
            return

        vt = self._infer(s.value)
        ct = self.ctype(vt)
        tmp = self._fresh('ret')
        val_c = self._emit_expr(s.value)
        self._w(f"{ct} {tmp} = {val_c};")

        # Retain ONLY if aliasing an existing variable (avoid over-retain leak)
        if vt[0] == 'string' and self._needs_retain(s.value):
            self._w(f"aura_retain_local({tmp});")
        elif vt[0] == 'slice' and self._needs_retain(s.value):
            self._w(f"aura_retain_local(({tmp}).data);")
        elif vt[0] == 'map' and self._needs_retain(s.value):
            self._w(f"aura_retain_local({tmp});")

        # Unwind all active scopes
        for dlist in reversed(self.defer_stack):
            for d in reversed(dlist): self._emit_stmt(d)
        for sc in reversed(self.arc_stack):
            for n, k in sc:
                if k == 'string': self._w(f"aura_release_local({n});")
                elif k == 'slice': self._w(f"aura_release_local(({n}).data);")
                elif k == 'map':   self._w(f"aura_release_local({n});")

        self._w(f"return {tmp};")

    def _emit_if(self, s: IfStmt):
        c = self._emit_expr(s.cond)
        self._w(f"if ({c}) {{")
        self._push_scope(); self.indent += 1
        for x in s.then.stmts: self._emit_stmt(x)
        self._close_scope(); self.indent -= 1; self._w('}')
        if s.els:
            self._w('else {'); self._push_scope(); self.indent += 1
            if isinstance(s.els, IfStmt): self._emit_if(s.els)
            else:
                for x in s.els.stmts: self._emit_stmt(x)
            self._close_scope(); self.indent -= 1; self._w('}')

    def _emit_for(self, s: ForStmt):
        self._push_scope()
        self._w('{'); self.indent += 1
        if s.init:
            if (isinstance(s.init, ExprStmt) and isinstance(s.init.expr, AssignExpr)
                and isinstance(s.init.expr.target, Ident)
                and getattr(s.init.expr.target, '_auto_declared', False)):
                nm = s.init.expr.target.name
                val_c = self._emit_expr(s.init.expr.value)
                self._w(f"int64_t {nm} = {val_c};")
                self._declare(nm, ('int',), None)
            else:
                self._emit_stmt(s.init)
        cond = self._emit_expr(s.cond) if s.cond else '1'
        self._w(f"for (; {cond}; ) {{")
        self.indent += 1
        self._push_scope()
        for x in s.body.stmts: self._emit_stmt(x)
        self._close_scope()
        if s.post: self._w(f"{self._emit_expr(s.post.expr)};")
        self.indent -= 1; self._w('}')
        self.indent -= 1; self._w('}')
        self._close_scope()

    def _emit_for_in(self, s: ForInStmt):
        self._push_scope()
        self._w('{'); self.indent += 1
        if s.kind == 'range':
            lo = self._fresh('lo'); hi = self._fresh('hi')
            self._w(f"int64_t {lo} = {self._emit_expr(s.iterable.lo)};")
            self._w(f"int64_t {hi} = {self._emit_expr(s.iterable.hi)};")
            op = '<=' if s.inclusive else '<'
            self._w(f"for (int64_t {s.var} = {lo}; {s.var} {op} {hi}; {s.var}++) {{")
            self.indent += 1
            self._push_scope()
            self._declare(s.var, ('int',), None)
            for x in s.body.stmts: self._emit_stmt(x)
            self._close_scope()
            self.indent -= 1; self._w('}')
        elif s.kind == 'slice':
            it = self._fresh('sl')
            et = getattr(s, '_resolved_iter')[1]
            ct = self.ctype(et)
            self._w(f"AuraSlice {it} = {self._emit_expr(s.iterable)};")
            self._w(f"for (int64_t __i = 0; __i < {it}.len; __i++) {{")
            self.indent += 1
            self._push_scope()
            self._w(f"{ct} {s.var} = (({ct}*)({it}).data)[__i];")
            self._declare(s.var, et, self.an._is_arc(et))
            for x in s.body.stmts: self._emit_stmt(x)
            self._close_scope()
            self.indent -= 1; self._w('}')
        elif s.kind == 'chan':
            it = self._fresh('ch')
            et = getattr(s, '_resolved_iter')[1]
            ct = self.ctype(et)
            self._w(f"AuraChan* {it} = {self._emit_expr(s.iterable)};")
            self._w(f"{ct} {s.var};")
            self._w(f"while (aura_chan_recv_ok({it}, &{s.var})) {{")
            self.indent += 1
            self._push_scope()
            self._declare(s.var, et, self.an._is_arc(et))
            for x in s.body.stmts: self._emit_stmt(x)
            self._close_scope()
            self.indent -= 1; self._w('}')
        self.indent -= 1; self._w('}')
        self._close_scope()

    def _emit_arena(self, s: ArenaStmt):
        self.arena_n += 1
        n = self.arena_n
        self._push_scope()
        self._w('{'); self.indent += 1
        self._w(f"AuraArena __arena_{n}; aura_arena_init(&__arena_{n}, 0);")
        self._w(f"AuraArena* __prev_{n} = __aura_cur_arena;")
        self._w(f"__aura_cur_arena = &__arena_{n};")
        for x in s.body.stmts: self._emit_stmt(x)
        self._close_scope()
        self._w(f"__aura_cur_arena = __prev_{n};")
        self._w(f"aura_arena_free(&__arena_{n});")
        self.indent -= 1; self._w('}')

    def _emit_spawn(self, s: SpawnStmt):
        call = s.call
        fname = call.func.name
        self.spawn_n += 1
        sid = self.spawn_n
        arg_exprs = [self._emit_expr(a) for a in call.args]
        sname = f"__aura_spawn_args_{sid}"
        tname = f"__aura_spawn_tramp_{sid}"
        lines = ["typedef struct {"]
        for i, a in enumerate(arg_exprs): lines.append(f"    __typeof__({a}) a{i};")
        lines.append(f"}} {sname};")
        body = [f"static void* {tname}(void* __p) {{", f"    {sname}* __a = ({sname}*)__p;"]
        args_str = ', '.join(f"__a->a{i}" for i in range(len(arg_exprs)))
        body.append(f"    {fname}({args_str});")
        body.append(f"    return NULL;\n}}")
        self.spawn_helpers.append('\n'.join(lines + body))
        self._w('{')
        self.indent += 1
        self._w(f"{sname} __args_{sid};")
        for i, a in enumerate(arg_exprs): self._w(f"__args_{sid}.a{i} = {a};")
        self._w(f"aura_spawn((AuraTaskFn){tname}, &__args_{sid}, sizeof(__args_{sid}));")
        self.indent -= 1
        self._w('}')

    def _infer(self, e: Node) -> Tuple:
        if isinstance(e, IntLit):    return ('int',)
        if isinstance(e, FloatLit):  return ('float',)
        if isinstance(e, BoolLit):   return ('bool',)
        if isinstance(e, StringLit): return ('string',)
        if isinstance(e, NilLit):    return ('ptr', ('void',))
        if isinstance(e, Ident):     return self.var_types.get(e.name, ('int',))
        if isinstance(e, CallExpr):  return getattr(e, '_resolved_ret', ('int',))
        if isinstance(e, ChanRecv):  return getattr(e, '_resolved_type', ('int',))
        if isinstance(e, MemberAccess): return getattr(e, '_resolved_type', ('int',))
        if isinstance(e, IndexExpr): return getattr(e, '_resolved_type', ('int',))
        if isinstance(e, SliceLit):  return ('slice', getattr(e, '_elem_type', ('int',)))
        if isinstance(e, StructLit): return ('struct', e.name)
        if isinstance(e, TryExpr):   return getattr(e, '_resolved_type', ('int',))
        if isinstance(e, BinaryExpr):
            if e.op in ('==','!=','<','>','<=','>=','&&','||','in'): return ('bool',)
            lt = self._infer(e.left)
            if lt == ('string',) or self._infer(e.right) == ('string',): return ('string',)
            return lt
        if isinstance(e, UnaryExpr):
            if e.op == '!': return ('bool',)
            if e.op == '&': return ('ptr', self._infer(e.operand))
            if e.op == '*': return self._infer(e.operand)[1]
            return self._infer(e.operand)
        if isinstance(e, AssignExpr): return self._infer(e.target)
        if isinstance(e, ChanMake):   return ('chan', getattr(e, '_elem_type', ('int',)))
        if isinstance(e, MapMake):    return ('map', getattr(e, '_key_type'), getattr(e, '_val_type'))
        return ('int',)

    def _emit_expr(self, e: Node) -> str:
        if isinstance(e, IntLit):    return f"INT64_C({e.value})"
        if isinstance(e, FloatLit):  return f"{e.value}"
        if isinstance(e, BoolLit):   return 'true' if e.value else 'false'
        if isinstance(e, StringLit):
            s = (e.value.replace('\\','\\\\').replace('"','\\"')
                 .replace('\n','\\n').replace('\t','\\t').replace('\r','\\r'))
            return f'aura_string_new("{s}")'
        if isinstance(e, NilLit):    return 'NULL'
        if isinstance(e, Ident):     return e.name

        if isinstance(e, BinaryExpr):
            l = self._emit_expr(e.left); r = self._emit_expr(e.right)
            lt, rt = self._infer(e.left), self._infer(e.right)
            if e.op == '+' and lt == ('string',) and rt == ('string',):
                return f"aura_string_concat({l}, {r})"
            if e.op in ('==','!=') and lt == ('string',) and rt == ('string',):
                return f"(aura_string_eq({l}, {r}) {e.op} true)"
            if e.op == 'in':
                return f"aura_map_contains({r}, (int64_t)({l}))"
            return f"({l} {e.op} {r})"

        if isinstance(e, UnaryExpr):
            if e.op == '*': return f"(*({self._emit_expr(e.operand)}))"
            return f"({e.op}{self._emit_expr(e.operand)})"

        if isinstance(e, AssignExpr):
            tgt_c = self._emit_expr(e.target)
            tt = self._infer(e.target)
            val_c = self._emit_expr(e.value)

            # Map indexing assignment: m[k] = v
            if isinstance(e.target, IndexExpr) and self._infer(e.target.obj)[0] == 'map':
                m_obj = self._emit_expr(e.target.obj)
                m_idx = self._emit_expr(e.target.index)
                return f"aura_map_set({m_obj}, (int64_t)({m_idx}), (int64_t)({val_c}))"

            if e.op == '=':
                if tt[0] == 'string' and isinstance(e.target, Ident):
                    t = self._fresh('old')
                    retain = f"aura_retain_local({val_c}); " if self._needs_retain(e.value) else ""
                    return (f"({{ AuraString* {t} = {tgt_c}; {retain}"
                            f"{tgt_c} = {val_c}; aura_release_local({t}); {tgt_c}; }})")
                return f"({tgt_c} = {val_c})"

            op = e.op[:-1]
            if op == '+' and tt == ('string',):
                concat = f"aura_string_concat({tgt_c}, {val_c})"
                t = self._fresh('old')
                return (f"({{ AuraString* {t} = {tgt_c}; {tgt_c} = {concat}; aura_release_local({t}); {tgt_c}; }})")
            return f"({tgt_c} {op}= {val_c})"

        if isinstance(e, CallExpr):
            intr = getattr(e, '_intrinsic', None)
            if intr in ('print','println'):
                a = e.args[0]; t = self._infer(a); c = self._emit_expr(a)
                if t == ('int',):    return f"aura_print_int({c})"
                if t == ('float',):  return f"aura_print_float({c})"
                if t == ('bool',):   return f"aura_print_bool({c})"
                if t == ('string',): return f"aura_print_str({c})"
                return f"aura_print_int((int64_t)({c}))"
            if intr == 'len':
                a = e.args[0]; t = self._infer(a); c = self._emit_expr(a)
                if t == ('string',): return f"((int64_t)({c})->len)"
                if t[0] == 'slice':  return f"((int64_t)({c}).len)"
                if t[0] == 'map':    return f"((int64_t)({c})->count)"
            if intr == 'ok':
                v = self._emit_expr(e.args[0])
                return f"((AuraResult){{ .ok = true, .err_code = 0, .val.i = (int64_t)({v}) }})"
            if intr == 'err':
                c = self._emit_expr(e.args[0])
                return f"((AuraResult){{ .ok = false, .err_code = (int64_t)({c}) }})"
            if intr == 'close':
                return f"(aura_chan_close({self._emit_expr(e.args[0])}), (void)0)"

            # Struct method calls with pointer-receiver auto-referencing
            if getattr(e, '_is_method', False):
                recv = self._emit_expr(e._method_recv)
                key = e._method_key
                st, mname = key.split('.', 1)
                fn = self.an.functions[key]
                if fn.receiver and fn.receiver.type.name == 'ptr' and not (self._infer(e._method_recv)[0] == 'ptr'):
                    recv = f"(&({recv}))"
                args_c = [recv] + [self._emit_expr(a) for a in e.args]
                return f"Aura_{st}_{mname}({', '.join(args_c)})"

            name = e.func.name
            args_c = [self._emit_expr(a) for a in e.args]
            return f"{name}({', '.join(args_c)})"

        if isinstance(e, ChanSend):
            ch = self._emit_expr(e.chan); val = self._emit_expr(e.value)
            return f"({{ __typeof__({val}) __sv = ({val}); aura_chan_send({ch}, &__sv); (void)0; }})"

        if isinstance(e, ChanRecv):
            ch = self._emit_expr(e.chan)
            ct = self.ctype(getattr(e, '_resolved_type', ('int',)))
            return f"({{ {ct} __rv; aura_chan_recv({ch}, &__rv); __rv; }})"

        if isinstance(e, ChanMake):
            return f"aura_chan_new(sizeof({self.ctype(getattr(e, '_elem_type'))}), (size_t)({self._emit_expr(e.capacity)}))"

        if isinstance(e, MapMake):
            sk = 'true' if getattr(e, '_key_type') == ('string',) else 'false'
            sv = 'true' if getattr(e, '_val_type') == ('string',) else 'false'
            return f"aura_map_new({sk}, {sv})"

        if isinstance(e, MemberAccess):
            obj = self._emit_expr(e.obj); ot = self._infer(e.obj)
            op = '->' if (ot[0] == 'ptr' and ot[1][0] == 'struct') else '.'
            return f"({obj}){op}{e.field}"

        if isinstance(e, IndexExpr):
            obj = self._emit_expr(e.obj); idx = self._emit_expr(e.index)
            ot = self._infer(e.obj)
            if ot[0] == 'slice':
                ct = self.ctype(getattr(e, '_resolved_type', ('int',)))
                return f"(({ct}*)(({obj}).data))[{idx}]"
            if ot[0] == 'map':
                ct = self.ctype(getattr(e, '_resolved_type', ('int',)))
                return f"(({ct})aura_map_get({obj}, (int64_t)({idx})))"
            return f"(({obj})[{idx}])"

        if isinstance(e, SliceLit):
            et = getattr(e, '_elem_type', ('int',)); ct = self.ctype(et); n = len(e.elements)
            buf = self._fresh('buf'); sl = self._fresh('sl')
            stmts = [f"{ct}* {buf} = ({ct}*)aura_alloc(sizeof({ct}) * {n});"]
            for i, el in enumerate(e.elements): stmts.append(f"{buf}[{i}] = {self._emit_expr(el)};")
            stmts.append(f"AuraSlice {sl} = {{ .data = {buf}, .len = {n}, .cap = {n} }};")
            return "({ " + " ".join(stmts) + f" {sl}; }})"

        if isinstance(e, StructLit):
            ctype = 'Aura_' + e.name
            provided = {fn: fe for fn, fe in e.inits}
            order = self.an.struct_order.get(e.name, [])
            parts = [f".{fn} = {self._emit_expr(provided[fn])}" for fn in order if fn in provided]
            return f"(({ctype}){{ {', '.join(parts)} }})"

        if isinstance(e, TryExpr):
            res = self._fresh('res')
            ct = self.ctype(getattr(e, '_resolved_type', ('int',)))
            call_c = self._emit_expr(e.expr)
            return (f"({{ AuraResult {res} = {call_c}; "
                    f"if (!{res}.ok) return {res}; "
                    f"({ct}){res}.val.i; }})")

        raise RuntimeError(f"codegen expr {type(e).__name__}")


# ═══════════════════════════════════════════════════════════════════════════
# 7 — DRIVER
# ═══════════════════════════════════════════════════════════════════════════

def format_error(err: AuraError, src: str) -> str:
    if isinstance(err, AuraError) and err.line > 0:
        lines = src.splitlines()
        if 0 < err.line <= len(lines):
            line = lines[err.line - 1]
            caret = ('\n  ' + ' ' * (err.col - 1) + '^') if err.col > 0 else ''
            return f"{err}\n  {err.line:>4} | {line}{caret}"
    return str(err)


def compile_aura(src: str, emit_only: bool = False, output: Optional[str] = None) -> str:
    try:
        toks = Lexer(src).tokenize()
        prog = Parser(toks).parse_program()
        an = Analyzer()
        an.analyze(prog)
        gen = CCodegen(an)
        c_src = gen.emit(prog)
    except AuraError as e:
        sys.stderr.write(format_error(e, src) + "\n")
        sys.exit(1)

    if emit_only: return c_src

    tmpdir = tempfile.mkdtemp(prefix='aura_')
    try:
        c_path = os.path.join(tmpdir, 'main.c')
        h_path = os.path.join(tmpdir, 'aura_rt.h')
        with open(c_path, 'w') as f: f.write(c_src)
        with open(h_path, 'w') as f: f.write(RUNTIME_HEADER)
        out_path = output or 'a.out'
        cc = os.environ.get('CC', 'cc')
        cmd = [cc, '-O2', '-std=c11', '-pthread', '-Wno-unused', c_path, '-o', out_path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            sys.stderr.write("C compiler error:\n" + r.stderr + "\n")
            sys.exit(2)
        return out_path
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    cmd, path = sys.argv[1], sys.argv[2]
    with open(path) as f: src = f.read()
    if cmd == 'emit':
        print(compile_aura(src, emit_only=True)); return
    out = sys.argv[sys.argv.index('-o') + 1] if '-o' in sys.argv else None
    if cmd == 'compile':
        b = compile_aura(src, output=out or os.path.splitext(path)[0])
        print(f"compiled -> {b}")
    elif cmd == 'run':
        b = compile_aura(src, output=out or os.path.join(tempfile.gettempdir(), 'aura_run'))
        r = subprocess.run([b]); sys.exit(r.returncode)


if __name__ == '__main__':
    main()
