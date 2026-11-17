/** Line icons drawn for BrainO (24-unit grid, stroke = current colour). */

import type { SVGProps } from "react";

type Props = SVGProps<SVGSVGElement> & { size?: number };

function Icon({ size = 20, children, ...rest }: Props) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...rest}
    >
      {children}
    </svg>
  );
}

export const Logo = (p: Props) => (
  <Icon size={22} stroke="var(--accent)" {...p}>
    <path d="M2 12h4l2-6 4 12 3-9 2 3h5" />
  </Icon>
);
export const Settings = (p: Props) => (
  <Icon {...p}>
    <path d="M4 6h9M17 6h3M4 12h3M11 12h9M4 18h11M19 18h1" />
    <circle cx="15" cy="6" r="2" />
    <circle cx="9" cy="12" r="2" />
    <circle cx="17" cy="18" r="2" />
  </Icon>
);
export const Folder = (p: Props) => (
  <Icon size={28} strokeWidth={1.8} stroke="var(--accent)" {...p}>
    <path d="M3 6h7l2 2h9v11H3z" />
  </Icon>
);
export const Download = (p: Props) => (
  <Icon size={28} strokeWidth={1.8} stroke="var(--accent)" {...p}>
    <path d="M12 4v11M7 10l5 5 5-5M4 20h16" />
  </Icon>
);
export const Wave = (p: Props) => (
  <Icon size={28} strokeWidth={1.8} stroke="var(--accent)" {...p}>
    <path d="M2 12h4l2-6 4 12 3-9 2 3h5" />
  </Icon>
);
export const Check = (p: Props) => (
  <Icon size={16} strokeWidth={3} {...p}>
    <path d="M5 12l5 5 9-10" />
  </Icon>
);
export const ChevronRight = (p: Props) => (
  <Icon size={18} {...p}>
    <path d="M9 6l6 6-6 6" />
  </Icon>
);
export const ChevronLeft = (p: Props) => (
  <Icon size={18} {...p}>
    <path d="M15 6l-6 6 6 6" />
  </Icon>
);
export const Close = (p: Props) => (
  <Icon size={16} {...p}>
    <path d="M6 6l12 12M18 6L6 18" />
  </Icon>
);
export const Up = (p: Props) => (
  <Icon size={16} {...p}>
    <path d="M12 19V5M6 11l6-6 6 6" />
  </Icon>
);
export const Down = (p: Props) => (
  <Icon size={16} {...p}>
    <path d="M12 5v14M6 13l6 6 6-6" />
  </Icon>
);
export const Plus = (p: Props) => (
  <Icon size={16} {...p}>
    <path d="M12 5v14M5 12h14" />
  </Icon>
);
