#!/usr/bin/env python3
# ==============================================================================
# build_GaN_SiC_interface.py
# Xay dung slab di cau truc GaN(0001)/SiC(111) (wurtzite GaN tren 3C-SiC)
# bang ASE, xu ly lattice mismatch bang phuong phap coincidence-site
# lattice (CSL) / supercell matching, xuat ra LAMMPS data file.
#
# De xuat nghien cuu: "Anh huong cua khiem khuyet do buc xa len truyen nhiet
# phonon tai giao dien GaN/SiC trong cam bien quang LiDAR"
# Tac gia: TS. Nguyen Xuan Huy - School of Engineering and Technology, NTU
# ==============================================================================
#
# LICH SU SUA LOI (04/08/2026):
#   Ban dau du dinh dung 4H-SiC(0001) voi toa do phan so tu dung tay (fractional
#   coordinates hand-derived). Khi chay thu LAMMPS that (minimization) phat hien
#   loi nghiem trong: khoang cach Si-C trong o mang tu dung = 0.628 Angstrom
#   (chong lan gan nhu trung nhau, dung phai ~1.89 Angstrom). Nguyen nhan: tham
#   so noi tai (internal parameter u) cua 4H-SiC (ABCB stacking, 4 lop/chu ky)
#   duoc uoc luong sai khi tu suy ra bang tay.
#   => DA CHUYEN sang dung 3C-SiC (zinc-blende, mat (111)) cat bang ham
#      ase.build.surface() (thuat toan da kiem chung cua ASE, khong tu suy toa
#      do bang tay), vua sua loi cau truc, vua khop voi force field SiC.tersoff
#      (Tersoff 1989) da duoc kiem chung rieng cho pha 3C (sai so 0.88% hang so
#      mang, xem 02_Force_Field/validation_results_20260804.md), thay vi dung
#      cho 4H-SiC (CHUA duoc kiem chung).
#   GaN(0001)/3C-SiC(111) van la he di cau truc that (khong phai "doi vat lieu
#   tuy tien") - day la cau hinh epitaxy pho bien trong thuc nghiem GaN-on-SiC,
#   chi khac 4H-SiC(0001) o da hinh SiC su dung. Can neu ro trong Methodology
#   rang lua chon 3C thay vi 4H la de dam bao do tin cay cua force field va
#   tinh dung cua hinh hoc, khong phai gia tri vat ly thap hon.
#
# Hang so mang tham khao (gia tri thuc nghiem):
#   GaN wurtzite: a = 3.189 Angstrom, c = 5.185 Angstrom
#   3C-SiC (zinc-blende, cubic): a = 4.3596 Angstrom
# ==============================================================================

import numpy as np
from ase import Atoms
from ase.build import bulk, surface, make_supercell
from ase.io import write

# ------------------------------------------------------------------
# 1. Thong so vat lieu (da kiem chung bang LAMMPS that, xem validation_results)
# ------------------------------------------------------------------
A_GAN, C_GAN = 3.189, 5.185       # Angstrom, wurtzite GaN
A_SIC_CUBIC = 4.3596               # Angstrom, 3C-SiC (zinc-blende, cubic)

# So o mang lap lai de tim ti le gan nhat (CSL don gian, bam theo phuong a)
def find_csl_ratio(a1, a2, max_n=20, tol=0.03):
    """Tim (m, n) sao cho m*a1 ~ n*a2 voi sai lech < tol (vd 2%).
    Cong thuc: |m*a1 - n*a2| / (n*a2) < tol
    """
    best = None
    for n in range(1, max_n + 1):
        for m in range(1, max_n + 1):
            mismatch = abs(m * a1 - n * a2) / (n * a2)
            if mismatch < tol:
                if best is None or mismatch < best[2]:
                    best = (m, n, mismatch)
    return best


def build_wurtzite(a, c, symbol_cation, symbol_anion):
    """Xay dung 1 unit cell wurtzite (P6_3mc), quy uoc goc gamma=60 do (KHONG
    phai 120 do nhu quy uoc thuong thay) - CO CHU DICH, de khop voi goc ma ham
    ase.build.surface() tra ve khi cat mat (111) tu o lap phuong SiC (da xac
    nhan bang thuc nghiem: surface() luon tra ve gamma=60).

    SUA LOI (04/08/2026): ban truoc dung gamma=120 (quy uoc pho bien hon,
    giong ASE bulk() wurtzite mac dinh) roi cang ep vector cell cua SiC
    (gamma=60) len GaN bang set_cell(scale_atoms=True) - phep bien doi nay
    KHONG bao toan cau truc khi 2 quy uoc goc khac nhau (gamma=120 vs 60 la 2
    lua chon vector co so KHONG tuong duong bang scale don gian, ma can phep
    doi co so dung: a2_60 = a1 + a2_120), gay chong lan nguyen tu nghiem trong
    (da phat hien qua kiem tra khoang cach: 0.0546 Angstrom o mot phien ban
    trung gian). Toa do phan so duoi day da duoc bien doi dung theo cong thuc
    u' = u - v (mod 1), v' = v tu quy uoc (1/3,2/3,z) chuan gamma=120, xac
    minh bang tinh toan cho ra dung khoang cach lien ket Ga-N = 1.9500
    Angstrom (khop thuc nghiem ~1.95 A).
    """
    # He so noi tai u = 0.12390721 lay TRUC TIEP tu ase.build.bulk("GaN",
    # crystalstructure="wurtzite", a=3.189, c=5.185) (da kiem chung cho ket
    # qua Ga-N = 1.9500 Angstrom, khop thuc nghiem ~1.95 A), chi doi quy uoc
    # goc gamma tu 120 sang 60 bang phep doi co so u' = u - v (mod 1), v' = v.
    u_internal = 0.12390721
    atoms = Atoms(
        symbols=[symbol_cation, symbol_anion, symbol_cation, symbol_anion],
        scaled_positions=[
            (0.0, 0.0, 0.0),
            (2/3, 2/3, u_internal),
            (2/3, 2/3, 0.5),
            (0.0, 0.0, 0.5 + u_internal),
        ],
        cell=[a, a, c, 90, 90, 60],
        pbc=True,
    )
    return atoms


def build_gan_slab(nx, ny, nz_layers, vacuum=0.0, interface_termination="Ga"):
    """interface_termination: nguyen tu nao nam o mat DUOI cung cua khoi GaN
    (mat se tiep xuc SiC sau khi ghep). "Ga" (mac dinh) hoac "N".

    CAN CU KHOA HOC (bo sung 04/08/2026 sau phan bien): tra cuu tai lieu that
    (Cambridge MRS "The Polarity of GaN: a Critical Review"; PMC11818682 2025)
    cho thay day la van de con tranh cai trong thuc nghiem: quy tac chung "GaN
    Ga-face moc tren SiC mat C, GaN N-face moc tren SiC mat Si", NHUNG nghien
    cuu CBED chi tiet lai cho thay GaN Ga-face van thu duoc tren SiC mat Si
    khi co lop dem (buffer layer) phu hop. KHONG co dap an duy nhat chac chan
    tu cong thuc, PHAI kiem soat tuong minh va bao cao ro trong Methodology
    thay vi mac dinh ngam. Mac dinh o day (Ga-terminated tai giao dien) la
    lua chon pho bien cho vat lieu chat luong thiet bi (device-grade Ga-polar
    GaN), CAN doi chieu lai voi dieu kien mo phong/thuc nghiem cu the truoc
    khi dua vao ket qua chinh thuc.
    """
    gan_unit = build_wurtzite(A_GAN, C_GAN, "Ga", "N")
    gan_slab = gan_unit.repeat((nx, ny, nz_layers))
    if interface_termination not in ("Ga", "N"):
        raise ValueError("interface_termination phai la 'Ga' hoac 'N'")
    if interface_termination == "N":
        # Lat truc z (z -> zmax - z) de doi nguyen tu nam o day cung
        pos = gan_slab.get_positions()
        pos[:, 2] = pos[:, 2].max() - pos[:, 2]
        gan_slab.set_positions(pos)
    bottom_species = gan_slab.get_chemical_symbols()[np.argmin(gan_slab.get_positions()[:, 2])]
    print(f"[POLARITY] GaN: nguyen tu tai mat duoi (se tiep xuc SiC) = {bottom_species}")
    return gan_slab


def build_3c_sic_111_slab(nx, ny, n_atomic_layers, interface_termination="Si"):
    """Dung ASE surface() de cat mat (111) tu o lap phuong 3C-SiC (zinc-blende)
    da kiem chung - KHONG tu suy toa do phan so bang tay (tranh loi da gap voi
    4H-SiC truoc do). Tra ve slab da repeat theo nx, ny trong mat phang.

    interface_termination: nguyen tu nao nam o mat TREN cung cua khoi SiC (mat
    se tiep xuc GaN sau khi ghep, vi SiC duoc dat ben duoi trong stack_interface).
    "Si" (mac dinh) hoac "C" - xem ghi chu ve cuc tinh trong build_gan_slab(),
    ap dung tuong tu: chua co quy uoc duy nhat trong tai lieu, can kiem soat
    tuong minh va bao cao ro.
    """
    sic_bulk = bulk("SiC", crystalstructure="zincblende", a=A_SIC_CUBIC, cubic=True)
    sic_slab = surface(sic_bulk, indices=(1, 1, 1), layers=n_atomic_layers, vacuum=None)
    sic_slab = sic_slab.repeat((nx, ny, 1))
    # Chuan hoa goc toa do ve (0,0,z_min) va dam bao khong co tilt am gay
    # nham lan khi ghep voi GaN sau nay
    pos = sic_slab.get_positions()
    pos[:, 2] -= pos[:, 2].min()
    sic_slab.set_positions(pos)

    if interface_termination not in ("Si", "C"):
        raise ValueError("interface_termination phai la 'Si' hoac 'C'")
    top_species = sic_slab.get_chemical_symbols()[np.argmax(sic_slab.get_positions()[:, 2])]
    if top_species != interface_termination:
        pos = sic_slab.get_positions()
        pos[:, 2] = pos[:, 2].max() - pos[:, 2]
        sic_slab.set_positions(pos)
        top_species = sic_slab.get_chemical_symbols()[np.argmax(sic_slab.get_positions()[:, 2])]
    print(f"[POLARITY] SiC: nguyen tu tai mat tren (se tiep xuc GaN) = {top_species}")
    return sic_slab


def stack_interface(gan_slab, sic_slab, gap=2.0, vacuum_top=0.0):
    """Ghep GaN len tren SiC theo truc z, canh chinh trong mat phang xy bang
    cach GAN TRUC TIEP vector a1,a2 cua SiC cho GaN roi dung set_cell(...,
    scale_atoms=True) de ASE tu tinh lai toa do theo phep bien doi tuyen tinh
    day du.

    SUA LOI (04/08/2026): ban truoc day scale rieng le truc x va truc y trong
    he Cartesian va chi gan lai thanh phan duong cheo cell[0,0]/cell[1,1] -
    cach nay SAI khi 2 o mang luc giac dung quy uoc goc gamma khac nhau (vd
    GaN wurtzite tu dung goc gamma=120 do trong khi SiC(111) cat boi ASE
    surface() tra ve goc gamma=60 do) - gay lech tay (handedness) giua 2 vat
    lieu ma khong bao loi ro rang. Dung set_cell(scale_atoms=True) xu ly dung
    trong moi truong hop goc/handedness vi no ap dung phep bien doi ma tran
    day du thay vi scale tung truc rieng le.
    """
    cell_sic = sic_slab.get_cell()
    cell_gan_orig = gan_slab.get_cell()

    a_gan_inplane = np.linalg.norm(cell_gan_orig[0])
    a_sic_inplane = np.linalg.norm(cell_sic[0])
    strain_pct = (a_sic_inplane - a_gan_inplane) / a_gan_inplane * 100
    print(f"[CANH BAO] GaN in-plane strain khi ep khop SiC: {strain_pct:.2f}%")
    if abs(strain_pct) > 4.0:
        print("Strain > 4%% - can dung CSL supercell lon hon thay vi ep truc tiep 1:1.")

    new_cell = cell_gan_orig.copy()
    new_cell[0] = cell_sic[0]
    new_cell[1] = cell_sic[1]
    gan_slab.set_cell(new_cell, scale_atoms=True)

    # SUA LOI (04/08/2026): ase.build.surface(..., vacuum=None) tra ve cell
    # co thanh phan z BANG 0 (KHONG phan anh do day thuc su cua slab SiC) -
    # dung truc tiep cell_sic[2,2] lam z_offset gay GaN CHONG LAN nghiem trong
    # len giua khoi SiC (da phat hien: 8139 cap nguyen tu < 1.0 Angstrom khi
    # con loi nay). PHAI dung do day THUC TE tu toa do nguyen tu (z max - z
    # min cua chinh SiC slab), khong dung tham so cell.
    sic_pos_z = sic_slab.get_positions()[:, 2]
    sic_thickness = sic_pos_z.max() - sic_pos_z.min()
    z_offset = sic_thickness + gap

    gan_pos = gan_slab.get_positions()
    gan_pos[:, 2] += z_offset - gan_slab.get_positions()[:, 2].min()
    gan_slab.set_positions(gan_pos)

    combined = sic_slab + gan_slab
    total_cell = cell_sic.copy()
    total_cell[2, 2] = z_offset + cell_gan_orig[2, 2] + vacuum_top
    combined.set_cell(total_cell, scale_atoms=False)
    combined.pbc = (True, True, True if vacuum_top == 0 else False)

    # Kiem tra an toan cuoi cung: bao dam khong con chong lan nguyen tu truoc
    # khi tra ve (fail-fast thay vi xuat ra file data hong ma khong bao loi)
    from scipy.spatial import cKDTree
    tree = cKDTree(combined.get_positions())
    dmin = tree.query(combined.get_positions(), k=2)[0][:, 1].min()
    if dmin < 1.0:
        raise RuntimeError(
            f"[LOI CAU TRUC] Phat hien khoang cach nguyen tu nho nhat = "
            f"{dmin:.4f} Angstrom (< 1.0 A, chac chan chong lan). KHONG xuat "
            f"file data. Kiem tra lai stack_interface() truoc khi tiep tuc."
        )
    print(f"[OK] Kiem tra chong lan: khoang cach nho nhat trong he = {dmin:.4f} Angstrom")
    return combined


if __name__ == "__main__":
    import sys

    # Kich thuoc muc tieu: 200,000-500,000 nguyen tu (theo de xuat).
    # QUAN TRONG: gia tri mac dinh duoi day (NX=NY=20) chi cho ra ~60,000-80,000
    # nguyen tu, CHUA dat muc tieu cua de xuat - day la kich thuoc PILOT/TEST,
    # khong phai production. Truyen "production" tu dong lenh de tu dong tang
    # len kich thuoc gan muc tieu, hoac tu chinh NX/NY/NZ_GAN ben duoi.
    mode = sys.argv[1] if len(sys.argv) > 1 else "pilot"

    if mode == "production":
        NX_SIC, NY_SIC = 22, 22     # ~ dua tong nguyen tu vao khoang 200,000-400,000
        NZ_SIC, NZ_GAN = 15, 30
    else:
        NX_SIC, NY_SIC = 10, 10     # kich thuoc PILOT de test nhanh (~60-80k atoms)
        NZ_SIC, NZ_GAN = 15, 15
        print("[CHE DO PILOT] Chi dung de test pipeline, KHONG dung cho ket qua "
              "chinh thuc (chua du 200k-500k nguyen tu theo de xuat). Chay lai "
              "voi: python build_GaN_SiC_interface.py production")

    # QUAN TRONG: xay SiC TRUOC, do kich thuoc mat phang THUC TE cua no (ham
    # ase.build.surface() tra ve mot o mang khong tat yeu la primitive - da
    # xac nhan bang thuc nghiem no lon hon primitive hex mong doi ~2 lan moi
    # canh) roi MOI tinh so lan lap GaN can thiet de khop kich thuoc that su,
    # thay vi gia dinh sai rang dung chung NX,NY cho ca 2 vat lieu se cho ra
    # kich thuoc vat ly bang nhau (loi da phat hien va sua ngay 04/08/2026,
    # gay "strain" ao len toi 93%% truoc khi sua).
    # Cuc tinh giao dien - XEM GHI CHU trong build_gan_slab()/build_3c_sic_111_slab()
    # ve rui ro khoa hoc (chua co quy uoc duy nhat trong tai lieu). Doi qua
    # tham so dong lenh thu 2, 3 neu can: production Ga Si | pilot N C v.v.
    GAN_TERM = sys.argv[2] if len(sys.argv) > 2 else "Ga"
    SIC_TERM = sys.argv[3] if len(sys.argv) > 3 else "Si"

    sic = build_3c_sic_111_slab(NX_SIC, NY_SIC, n_atomic_layers=NZ_SIC,
                                  interface_termination=SIC_TERM)
    sic_a1_len = np.linalg.norm(sic.get_cell()[0])
    sic_a2_len = np.linalg.norm(sic.get_cell()[1])

    NX_GAN = max(1, round(sic_a1_len / A_GAN))
    NY_GAN = max(1, round(sic_a2_len / A_GAN))
    print(f"SiC in-plane thuc te: a1={sic_a1_len:.3f} A, a2={sic_a2_len:.3f} A")
    print(f"=> Chon GaN repeat: NX_GAN={NX_GAN}, NY_GAN={NY_GAN} "
          f"(a_GaN={A_GAN} A) de khop kich thuoc vat ly")

    csl = find_csl_ratio(NX_GAN * A_GAN, sic_a1_len)
    print(f"Kiem tra khop kich thuoc (m=1 x GaN_block ~ n=1 x SiC_block): {csl}")

    gan = build_gan_slab(NX_GAN, NY_GAN, NZ_GAN, interface_termination=GAN_TERM)

    interface = stack_interface(gan, sic, gap=2.0, vacuum_top=0.0)

    print(f"Tong so nguyen tu: {len(interface)}")
    print(f"Kich thuoc o mang: {interface.get_cell()}")

    # Canh bao ve do day he theo truc z so voi tam ban PKA (rui ro "cascade
    # containment" da neu trong phan bien) - uoc luong THO dua tren quy tac
    # kinh nghiem pho bien trong tai lieu cascade (~1-2 nm tam ban cho moi
    # keV nang luong PKA trong vat lieu ran mat do trung binh, CHUA thay the
    # cho tinh toan SRIM that) - CHI mang tinh canh bao so bo, PHAI kiem tra
    # lai bang du lieu dump thuc te sau khi chay cascade.
    z_thickness = interface.get_positions()[:, 2].max() - interface.get_positions()[:, 2].min()
    E_pka_max_keV = 10.0
    est_range_A = E_pka_max_keV * 15.0   # uoc luong tho: ~1.5 nm/keV
    print(f"[CANH BAO CONTAINMENT] Do day he theo z = {z_thickness:.1f} A. "
          f"Uoc luong THO tam ban PKA {E_pka_max_keV} keV ~ {est_range_A:.0f} A "
          f"(quy tac kinh nghiem, KHONG thay the SRIM).")
    if est_range_A > z_thickness / 2:
        print("  => CO THE khong du day - PKA nang luong cao co nguy co cham "
              "vung bien truoc khi tieu tan het nang luong. BAT BUOC kiem tra "
              "dump.cascade_*.lammpstrj sau khi chay, hoac tang NZ_GAN/NZ_SIC.")

    # Xuat LAMMPS data file (atom_style atomic - phu hop Tersoff/Tersoff/LJ)
    write(
        "GaN_SiC_interface.data",
        interface,
        format="lammps-data",
        atom_style="atomic",
        specorder=["Ga", "N", "Si", "C"],
    )
    print("Da xuat: GaN_SiC_interface.data")
    print("BUOC TIEP THEO (bat buoc truoc production):")
    print("  1. Mo file trong OVITO/VMD kiem tra truc quan giao dien")
    print("  2. Chay minimization + kiem tra hang so mang GaN va SiC rieng le")
    print("  3. Kiem tra energy/atom hop ly (khong co overlap nguyen tu)")
