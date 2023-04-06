function [p1, p2] = plotSurfaces(label1, label2)
    figure
    s1 = isosurface(label1);
    p1 = patch(s1);
    view(3)
    set(p1,'FaceColor',[0.5 1 0.5]);
    set(p1, 'EdgeColor','none');
    set(p1, 'FaceAlpha', 0.7);
    camlight;
    lighting gouraud;
    hold on;
    s2 = isosurface(label2);
    p2 = patch(s2);
    set(p2,'FaceColor',[0.6350 0.0780 0.1840]);
    set(p2, 'EdgeColor','none');
    set(p2, 'FaceAlpha', 0.7);
end