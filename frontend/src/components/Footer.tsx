export function Footer() {
  return (
    <footer className="border-t border-border py-8 text-center text-sm text-muted">
      <p>
        Causality &mdash; causal machine learning for breast cancer diagnosis. Built for education and
        research; not a medical device.
      </p>
      <p className="mt-1">
        <a
          href="https://github.com/Desmondonam/Causality"
          className="underline decoration-dotted underline-offset-2 hover:text-foreground"
        >
          Source on GitHub
        </a>
      </p>
    </footer>
  );
}
